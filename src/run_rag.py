import os
import sys
import torch

# Add src to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_loader import load_pubmedqa_dataset, sample_dataset
from baseline_eval import load_model
from rag_eval import build_bm25_index, evaluate_rag

def main():
    # Configurations — all controllable via env vars
    MODEL_NAME  = os.getenv("MODEL_NAME",  "Qwen/Qwen2.5-3B-Instruct")
    NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", 200))
    BATCH_SIZE  = int(os.getenv("BATCH_SIZE",  8))
    TOP_K       = int(os.getenv("TOP_K",       3))

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_csv  = os.path.join(
        current_dir,
        f'pubmedqa_rag_results_{MODEL_NAME.replace("/", "_")}.csv'
    )

    # ── Step 1: Load PubMedQA ─────────────────────────────────────────────────
    print(f"Loading PubMedQA dataset...")
    pubmedqa_full    = load_pubmedqa_dataset()
    pubmedqa_sampled = sample_dataset(pubmedqa_full, num_samples=NUM_SAMPLES)
    print(f"Sampled {len(pubmedqa_sampled)} examples for evaluation.")

    # ── Step 2: Build BM25 index over full dataset ────────────────────────────
    print(f"Building BM25 index over full PubMedQA corpus...")
    bm25, corpus, corpus_meta = build_bm25_index(pubmedqa_full)
    print(f"Index built: {len(corpus)} paragraphs indexed.")

    # ── Step 3: Load model ────────────────────────────────────────────────────
    print(f"Loading model: {MODEL_NAME}...")
    tokenizer, model = load_model(MODEL_NAME)
    print(f"Model loaded on: {next(model.parameters()).device}")

    # ── Step 4: Run RAG evaluation ────────────────────────────────────────────
    print(f"\nStarting RAG evaluation (top_k={TOP_K}, samples={NUM_SAMPLES})...")
    print(f"Results will be saved to: {output_csv}\n")

    results, saved_path = evaluate_rag(
        dataset    = pubmedqa_sampled,
        tokenizer  = tokenizer,
        model      = model,
        bm25       = bm25,
        corpus     = corpus,
        batch_size = BATCH_SIZE,
        output_csv = output_csv,
        k          = TOP_K
    )

    # ── Step 5: Print summary ─────────────────────────────────────────────────
    total      = results['total']
    correct    = results['correct']
    unknown    = results['unknown']
    faithful   = results['faithful']
    unfaithful = results['unfaithful']

    hallucination_rate = (total - correct) / total if total > 0 else 0
    accuracy           = correct / total if total > 0 else 0

    print(f"\n{'='*50}")
    print(f"RAG Evaluation Complete — {MODEL_NAME}")
    print(f"{'='*50}")
    print(f"Total Questions    : {total}")
    print(f"Correct            : {correct}")
    print(f"Accuracy           : {accuracy:.2%}")
    print(f"Hallucination Rate : {hallucination_rate:.2%}")
    print(f"Unknown predictions: {unknown}")
    if faithful + unfaithful > 0:
        print(f"Faithful outputs   : {faithful}")
        print(f"Unfaithful outputs : {unfaithful}")
        print(f"Faithfulness Rate  : {faithful / (faithful + unfaithful):.2%}")
    print(f"Results saved to   : {saved_path}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()
