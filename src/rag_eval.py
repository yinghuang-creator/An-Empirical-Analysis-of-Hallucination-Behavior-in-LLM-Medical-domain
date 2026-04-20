from rank_bm25 import BM25Okapi
from tqdm import tqdm
import torch
import csv
import os
import re

# ── Build BM25 index from PubMedQA corpus ──────────────────────────────────────
def build_bm25_index(dataset):
    """
    Flatten all CONTEXTS paragraphs from PubMedQA into a single corpus.
    Returns the BM25 index and the list of (doc_text, source_question) tuples.
    """
    corpus = []
    corpus_meta = []  # track which record each paragraph came from

    for record in dataset:
        for paragraph in record['CONTEXTS']:
            corpus.append(paragraph)
            corpus_meta.append({
                'question': record['QUESTION'],
                'final_decision': record['final_decision']
            })

    # Tokenize by whitespace for BM25
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25, corpus, corpus_meta


# ── Retrieve top-k abstracts for a query ───────────────────────────────────────
def retrieve_top_k(query, bm25, corpus, k=3):
    """
    Given a question, retrieve the top-k most relevant paragraphs using BM25.
    """
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    top_k_docs = [corpus[i] for i in top_k_indices]
    return top_k_docs


# ── Build RAG prompt ───────────────────────────────────────────────────────────
def build_rag_prompt(question, retrieved_docs):
    """
    Inject retrieved abstracts as context into the prompt.
    Fixed template as specified in the proposal.
    """
    context = "\n\n".join([f"[Abstract {i+1}]: {doc}" for i, doc in enumerate(retrieved_docs)])

    prompt = (
        "You are a medical expert. Read the context carefully and answer ONLY based on it.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer with exactly one word - yes, no, or maybe:"
    )
    return prompt


# ── Load model ────────────────────────────────────────────────────────────────
def load_model(model_name):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    shared_kwargs = {"token": hf_token} if hf_token else {}

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", **shared_kwargs
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        **shared_kwargs,
    )
    return tokenizer, model


# ── Generate answers in batches ────────────────────────────────────────────────
def generate_answers_batch(prompts, tokenizer, model, max_new_tokens=50):
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0
        )

    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [resp[len(prompt):].strip() for prompt, resp in zip(prompts, responses)]


# ── Extract yes/no/maybe prediction ───────────────────────────────────────────
def extract_decision(text):
    text_lower = text.lower().strip()
    
    # Check for "Answer: Yes/No/Maybe" pattern
    import re
    match = re.search(r'answer[:\s]+\s*(yes|no|maybe)', text_lower)
    if match:
        return match.group(1)
    
    for label in ['yes', 'no', 'maybe']:
        if text_lower.startswith(label):
            return label
    
    first_chunk = text_lower[:50]
    for label in ['yes', 'no', 'maybe']:
        if label in first_chunk:
            return label
    
    for label in ['yes', 'no', 'maybe']:
        if label in text_lower:
            return label
    
    return 'unknown'


# ── MiniCheck faithfulness scoring ────────────────────────────────────────────
def load_minicheck():
    """
    Load MiniCheck for faithfulness scoring.
    Falls back gracefully if not installed.
    """
    try:
        from minicheck.minicheck import MiniCheck
        scorer = MiniCheck(model_name='roberta-large', device='cuda' if torch.cuda.is_available() else 'cpu')
        return scorer
    except ImportError:
        print("MiniCheck not installed. Skipping faithfulness scoring.")
        print("Install with: pip install minicheck-factchecker")
        return None


def score_faithfulness(scorer, documents, claims):
    """
    Score whether each claim is faithful to its corresponding document.
    Returns list of scores (1=faithful, 0=unfaithful).
    """
    if scorer is None:
        return [None] * len(claims)

    try:
        pred_labels, scores, _, _ = scorer.score(
            docs=documents,
            claims=claims
        )
        return scores
    except Exception as e:
        print(f"MiniCheck scoring failed: {e}")
        return [None] * len(claims)


# ── Main RAG evaluation ────────────────────────────────────────────────────────
def evaluate_rag(dataset, tokenizer, model, bm25, corpus,
                 batch_size=8, output_csv='rag_results.csv', k=3):
    """
    Main RAG evaluation loop.
    For each PubMedQA question:
      1. Retrieve top-k abstracts via BM25
      2. Build RAG prompt
      3. Generate answer
      4. Score faithfulness with MiniCheck
      5. Compare against gold label
    """
    scorer = load_minicheck()

    stats = {
        'total': 0,
        'correct': 0,
        'unknown': 0,
        'faithful': 0,
        'unfaithful': 0
    }

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, output_csv)

    with open(output_path, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = [
            'question_id',
            'question',
            'gold_label',
            'prediction',
            'is_correct',
            'faithfulness_score',
            'retrieved_context',
            'model_output'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for i in tqdm(range(0, len(dataset), batch_size), desc="RAG Evaluation"):
            batch = [dataset[idx] for idx in range(i, min(i + batch_size, len(dataset)))]

            # Step 1: Retrieve top-k abstracts for each question
            batch_retrieved = [
                retrieve_top_k(ex['QUESTION'], bm25, corpus, k=k)
                for ex in batch
            ]

            # Step 2: Build RAG prompts
            prompts = [
                build_rag_prompt(ex['QUESTION'], retrieved_docs)
                for ex, retrieved_docs in zip(batch, batch_retrieved)
            ]

            # Step 3: Generate answers
            outputs = generate_answers_batch(prompts, tokenizer, model)

            # Step 4: Score faithfulness with MiniCheck
            contexts_for_scoring = [
                " ".join(docs) for docs in batch_retrieved
            ]
            faithfulness_scores = score_faithfulness(scorer, contexts_for_scoring, outputs)

            # Step 5: Save results
            for j in range(len(batch)):
                ex = batch[j]
                gold = ex['final_decision'].strip().lower()
                pred = extract_decision(outputs[j])
                is_correct = (pred == gold)
                faith_score = faithfulness_scores[j]

                writer.writerow({
                    'question_id': i + j,
                    'question': ex['QUESTION'],
                    'gold_label': gold,
                    'prediction': pred,
                    'is_correct': is_correct,
                    'faithfulness_score': faith_score,
                    'retrieved_context': " | ".join(batch_retrieved[j]),
                    'model_output': outputs[j]
                })

                stats['total'] += 1
                if is_correct:
                    stats['correct'] += 1
                if pred == 'unknown':
                    stats['unknown'] += 1
                if faith_score is not None:
                    if faith_score >= 0.5:
                        stats['faithful'] += 1
                    else:
                        stats['unfaithful'] += 1

            csv_file.flush()

    return stats, output_path
