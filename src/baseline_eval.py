from datasets import load_dataset
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import csv

def load_medqa_sample(split='test', sample_size=200, seed=42):
    dataset = load_dataset("bigbio/med_qa", "med_qa_en_4options_source", split=split)
    random.seed(seed)
    sampled_indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
    return dataset.select(sampled_indices)

def build_medqa_prompt(example):
    question = example['question']
    options = example['options']
    option_lines = []
    letters = ['A', 'B', 'C', 'D', 'E', 'F']
    for i, option in enumerate(options):
        option_lines.append(f"{letters[i]}. {option}")
    
    prompt = (
        "You are a medical expert. Please answer the following multiple-choice question:\n\n"
        f"Question: {question}\n"
        f"Options:\n" + "\n".join(option_lines) + "\n"
        "Answer (choose the correct option letter):"
    )
    return prompt

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map = "auto")
    return tokenizer, model

def generate_answer(prompt, tokenizer, model, max_new_tokens=10):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer[len(prompt):].strip()

def extract_answer_letter(answer):
    output_text = answer.strip().upper()

    if output_text in ['A', 'B', 'C', 'D', 'E', 'F']:
        return output_text
    
    match = re.search(r'\b([A-F])\b', output_text)
    if match:
        return match.group(1)
    return None

def evaluate_medqa_baseline(dataset, tokenizer, model, output_csv='medqa_baseline_results.csv'):
    total = 0
    correct = 0
    failed_to_extract = 0
    
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['question_id', 'prompt', 'model_output', 'predicted_answer', 'correct_answer', 'is_correct'])
        writer.writeheader()

        for idx, example in enumerate(dataset):
            prompt = build_medqa_prompt(example)
            model_output = generate_answer(prompt, tokenizer, model)
            predicted_answer = extract_answer_letter(model_output)
            correct_answer = str(example['answer']).strip().upper()
            is_correct = (predicted_answer == correct_answer)

            writer.writerow({
                'question_id': idx,
                'prompt': prompt,
                'model_output': model_output,
                'predicted_answer': predicted_answer,
                'correct_answer': correct_answer,
                'is_correct': is_correct
            })

            total += 1
            if is_correct:
                correct += 1
            if predicted_answer is None:
                failed_to_extract += 1
        
        accuracy = correct / total if total > 0 else 0.0
        hallucination_rate = 1.0 - accuracy

        return {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "hallucination_rate": hallucination_rate,
            "failed_to_extract": failed_to_extract
        }