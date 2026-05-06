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
    option_lines = [f"{opt['key']}. {opt['value']}" for opt in options]
    
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
    option_lines = [f"{opt['key']}. {opt['value']}" for opt in options]
    
    prompt = (
        "You are a medical expert.\n"
        "Solve the following multiple-choice medical question.\n"
        "Think briefly and carefully.\n"
        "You must follow this exact format:\n"
        "REASONING: <brief reasoning>\n"
        "FINAL_ANSWER: <A/B/C/D>\n"
        "Do not output anything after FINAL_ANSWER.\n\n"
        f"Question: {question}\n"
        f"Options:\n" + "\n".join(option_lines) + "\n\n"
        "REASONING:"
    )
    return prompt

def build_answer_only_recovery_prompt(example, cot_output):
    """Build recovery prompt for parse-failed CoT outputs."""
    question = example["question"]
    options = example["options"]
    option_lines = [f"{opt['key']}. {opt['value']}" for opt in options]

    prompt = (
        "You are given a medical multiple-choice question and a model's reasoning.\n"
        "Your task is to output only the final correct option letter.\n"
        "Output exactly one capital letter: A, B, C, or D.\n\n"
        f"Question:\n{question}\n"
        f"Options:\n" + "\n".join(option_lines) + "\n\n"
        f"Reasoning:\n{cot_output}\n\n"
        "Final letter:"
    )
    return prompt

def load_model(model_name):
    """Load tokenizer and model with optional HF token."""
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    shared_kwargs = {"token": hf_token} if hf_token else {}

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            **shared_kwargs
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

def format_as_chat(prompt, tokenizer):
    """Apply chat template when available."""
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    return prompt

#Generate outputs in batch and decode only newly generated tokens.
def generate_answers_batch(prompts, tokenizer, model, max_new_tokens):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3
        )

    # Decode only generated continuation, not the full input prompt
    prompt_len = inputs["input_ids"].shape[1]
    gen_tokens = outputs[:, prompt_len:]
    responses = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

    return [resp.strip() for resp in responses]


def extract_answer_letter(text, mode="baseline"):
    text = (text or "").strip()

    if not text:
        return None

    if mode == "cot":
        patterns = [
            r"FINAL_ANSWER\s*:\s*([A-D])\s*$",
            r"Therefore,\s*the correct answer is\s*\[?([A-D])\]?\s*$",
            r"Answer\s*:\s*([A-D])\s*$",
            r"^\s*([A-D])\s*$",
            r"^\s*([A-D])[.)]?\s*$",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()

        tail = text[-80:]
        tail_match = re.search(r"\b([A-D])\b", tail, flags=re.IGNORECASE)
        if tail_match:
            return tail_match.group(1).upper()

        return None

    else:
        # Restore original baseline parser
        text_clean = text.strip().upper()
        if text_clean and text_clean[0] in ["A", "B", "C", "D"]:
            return text_clean[0]

        match = re.search(r"\b([A-D])\b", text_clean)
        return match.group(1) if match else None

def normalize_gold_answer(example):
    """
    Normalize gold label to A/B/C/D.
    This function is written defensively because datasets may store labels
    as letters, integers, or numeric strings.
    """
    raw = example.get("answer_idx", None)

    # Already letter format
    if raw in ["A", "B", "C", "D"]:
        return raw

    # Integer format
    if isinstance(raw, int):
        mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
        return mapping.get(raw, "UNKNOWN")

    # String format
    raw_str = str(raw).strip().upper()

    if raw_str in ["A", "B", "C", "D"]:
        return raw_str

    if raw_str in ["0", "1", "2", "3"]:
        mapping = {"0": "A", "1": "B", "2": "C", "3": "D"}
        return mapping[raw_str]

    return "UNKNOWN"

def recover_failed_cot_answer(example, cot_output, tokenizer, model):
    """
    Run a second-pass recovery prompt when CoT parsing fails.
    """
    recovery_prompt = build_answer_only_recovery_prompt(example, cot_output)
    recovery_prompt = format_as_chat(recovery_prompt, tokenizer)

    recovery_output = generate_answers_batch(
        [recovery_prompt],
        tokenizer,
        model,
        max_new_tokens=5
    )[0]

    recovered_pred = extract_answer_letter(recovery_output, mode="baseline")
    return recovered_pred, recovery_output


def evaluate_combined_batch(
    dataset,
    tokenizer,
    model,
    batch_size=8,
    output_csv="medqa_combined_results.csv"
):
    """
    Evaluate baseline and CoT on the same MedQA subset.
    Also tracks parse failures and CoT recovery behavior.
    """
    stats = {
        "baseline_correct": 0,
        "cot_correct": 0,
        "baseline_parse_fail": 0,
        "cot_parse_fail": 0,
        "cot_recovered": 0,
        "total": 0
    }

    with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "question_id",
            "correct_answer",
            "baseline_prediction",
            "baseline_is_correct",
            "baseline_output",
            "cot_prediction",
            "cot_is_correct",
            "cot_output",
            "cot_recovery_output"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating Batches"):
            batch_examples = [
                dataset[idx]
                for idx in range(i, min(i + batch_size, len(dataset)))
            ]

            # Build baseline prompts
            baseline_prompts = [build_medqa_baseline_prompt(ex) for ex in batch_examples]

            # Build CoT prompts
            cot_prompts = [
                format_as_chat(build_medqa_cot_prompt(ex), tokenizer)
                for ex in batch_examples
            ]

            # Generate baseline outputs
            baseline_outputs = generate_answers_batch(
                baseline_prompts,
                tokenizer,
                model,
                max_new_tokens=10
            )

            # Generate CoT outputs
            cot_outputs = generate_answers_batch(
                cot_prompts,
                tokenizer,
                model,
                max_new_tokens=128
            )

            # Save results
            for j, ex in enumerate(batch_examples):
                correct_letter = normalize_gold_answer(ex)

                # ----- Baseline -----
                base_pred = extract_answer_letter(
                    baseline_outputs[j],
                    mode="baseline"
                )
                if base_pred is None:
                    stats["baseline_parse_fail"] += 1

                base_correct = (base_pred == correct_letter)

                # ----- CoT -----
                cot_pred = extract_answer_letter(
                    cot_outputs[j],
                    mode="cot"
                )
                cot_recovery_output = ""

                if cot_pred is None:
                    stats["cot_parse_fail"] += 1
                    recovered_pred, cot_recovery_output = recover_failed_cot_answer(
                        ex,
                        cot_outputs[j],
                        tokenizer,
                        model
                    )

                    if recovered_pred is not None:
                        cot_pred = recovered_pred
                        stats["cot_recovered"] += 1

                cot_correct = (cot_pred == correct_letter)

                writer.writerow({
                    "question_id": i + j,
                    "correct_answer": correct_letter,
                    "baseline_prediction": base_pred,
                    "baseline_is_correct": base_correct,
                    "baseline_output": baseline_outputs[j],
                    "cot_prediction": cot_pred,
                    "cot_is_correct": cot_correct,
                    "cot_output": cot_outputs[j],
                    "cot_recovery_output": cot_recovery_output
                })

                stats["total"] += 1
                if base_correct: stats["baseline_correct"] += 1
                if cot_correct: stats["cot_correct"] += 1
            
            csv_file.flush()

    return stats