from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import csv
import os
#Standard (Zero-shot)
def build_medqa_baseline_prompt(example):
    question = example['question']
    options = example['options']
    option_lines = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]
    
    prompt = (
        "You are a medical expert. Please answer the following multiple-choice question:\n\n"
        f"Question: {question}\n"
        f"Options:\n" + "\n".join(option_lines) + "\n"
        "Answer (choose the correct option letter):"
    )
    return prompt
#Chain-of-Thought (CoT)
def build_medqa_cot_prompt(example):
    question = example['question']
    options = example['options']
    option_lines = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]
    
    prompt = (
        "You are a medical expert. Please answer the following multiple-choice question.\n"
        "First, think step-by-step and explain your diagnostic reasoning.\n"
        "Finally, conclude your response with the exact phrase: 'Therefore, the correct answer is [Letter]'.\n\n"
        f"Question: {question}\n"
        f"Options:\n" + "\n".join(option_lines) + "\n\n"
        "Reasoning and Answer:"
    )
    return prompt

def load_model(model_name):
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    shared_kwargs = {"token": hf_token} if hf_token else {}

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", **shared_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            **shared_kwargs,
        )
        return tokenizer, model
    except OSError as e:
        message = str(e).lower()
        if "gated repo" in message or "403" in message or "not in the authorized list" in message:
            raise RuntimeError(
                f"Cannot access model '{model_name}'. It appears to be gated/private. "
                "Either: (1) request access and login with 'huggingface-cli login', "
                "or (2) set a public model via MODEL_NAME env var, e.g. "
                "MODEL_NAME=Qwen/Qwen2.5-3B-Instruct. "
                "If you already have access, set HF_TOKEN/HUGGINGFACE_TOKEN in your environment."
            ) from e
        raise

def generate_answers_batch(prompts, tokenizer, model, max_new_tokens):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False, 
            temperature=0.0
        )
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [resp[len(prompt):].strip() for prompt, resp in zip(prompts, responses)]

def extract_answer_letter(text, mode="baseline"):
    if mode == "cot":
        match = re.search(r"correct answer is \[?([A-D])\]?", text, re.IGNORECASE)
        if match: return match.group(1).upper()
        matches = re.findall(r'\b([A-D])\b', text.upper())
        return matches[-1] if matches else None
    else:
        # For baseline, the output should just be the letter
        text_clean = text.strip().upper()
        if text_clean and text_clean[0] in ['A', 'B', 'C', 'D']: return text_clean[0]
        match = re.search(r'\b([A-D])\b', text_clean)
        return match.group(1) if match else None

def evaluate_combined_batch(dataset, tokenizer, model, batch_size=8, output_csv='medqa_combined_results.csv'):
    stats = {"baseline_correct": 0, "cot_correct": 0, "total": 0}
    
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = [
            'question_id', 'correct_answer', 
            'baseline_prediction', 'baseline_is_correct', 'baseline_output',
            'cot_prediction', 'cot_is_correct', 'cot_output'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating Batches"):
            batch_examples = [dataset[idx] for idx in range(i, min(i + batch_size, len(dataset)))]
            
            # Baseline
            baseline_prompts = [build_medqa_baseline_prompt(ex) for ex in batch_examples]
            baseline_outputs = generate_answers_batch(baseline_prompts, tokenizer, model, max_new_tokens=10)
            
            # CoT
            cot_prompts = [build_medqa_cot_prompt(ex) for ex in batch_examples]
            cot_outputs = generate_answers_batch(cot_prompts, tokenizer, model, max_new_tokens=256)
            
            # Save Results
            for j in range(len(batch_examples)):
                ex = batch_examples[j]
                
                correct_ans_text = str(ex['answer']).strip()
                correct_letter = "UNKNOWN"
                for opt_idx, opt_text in enumerate(ex['options']):
                    if str(opt_text).strip() == correct_ans_text:
                        correct_letter = chr(65 + opt_idx) # A, B, C, D...
                        break
                
                # get baseline prediction
                base_pred = extract_answer_letter(baseline_outputs[j], mode="baseline")
                base_correct = (base_pred == correct_letter)
                
                cot_pred = extract_answer_letter(cot_outputs[j], mode="cot")
                cot_correct = (cot_pred == correct_letter)

                writer.writerow({
                    'question_id': i + j,
                    'correct_answer': correct_letter,
                    'baseline_prediction': base_pred,
                    'baseline_is_correct': base_correct,
                    'baseline_output': baseline_outputs[j],
                    'cot_prediction': cot_pred,
                    'cot_is_correct': cot_correct,
                    'cot_output': cot_outputs[j]
                })

                stats["total"] += 1
                if base_correct: stats["baseline_correct"] += 1
                if cot_correct: stats["cot_correct"] += 1
            
            csv_file.flush()
        
        return stats