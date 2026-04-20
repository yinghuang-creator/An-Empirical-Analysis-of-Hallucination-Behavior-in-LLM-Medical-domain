"""
inference.py
Unified model wrapper for zero-shot, CoT, and RAG conditions.
Produces raw text output + extracted answer letter for evaluation.
"""

import re
import json
from dataclasses import dataclass, field
from typing import Optional, Literal
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from load_dataset import QASample

Condition = Literal["zero_shot", "cot", "rag", "sft", "sft_rag"]

# ── Output schema ───────────────────────────────────────────────────────────
@dataclass
class InferenceResult:
    sample_id: str
    model_name: str
    condition: Condition
    raw_output: str
    extracted_answer: Optional[str]  # letter for MedQA; yes/no/maybe for PubMedQA
    is_hallucination: Optional[bool] = field(default=None)
    prompt_used: str = field(default="", repr=False)


# ── Prompt builders ─────────────────────────────────────────────────────────
def build_prompt_medqa(
    sample: QASample,
    condition: Condition,
    context: Optional[str] = None,
) -> str:
    choices_block = "\n".join(
        f"{choice['key']}. {choice['value']}" for choice in sample.choices
    )

    if condition == "cot":
        instruction = (
            "Think step by step before giving your final answer. "
            "End your response with 'Answer: <letter>'."
        )
        suffix = "\n\nAnswer: "
    else:
        instruction = "Answer with only the letter of the correct option (A, B, C, or D)."
        suffix = "\n\nAnswer: "

    if condition == "rag" and context:
        rag_block = f"Context:\n{context}\n\n"
    else:
        rag_block = ""

    return (
        f"{rag_block}"
        f"Question: {sample.question}\n\n"
        f"{choices_block}\n\n"
        f"{instruction}"
        f"{suffix}"
    )


def build_prompt_pubmedqa(
    sample: QASample,
    condition: Condition,
    context: Optional[str] = None,
) -> str:
    if context:
        return (
            f"Context:\n{context}\n\n"
            f"Question: {sample.question}\n\n"
            "Answer with only 'yes', 'no', or 'maybe'.\n\n"
            "The correct answer is"
        )
    else:
        return (
            f"Question: {sample.question}\n\n"
            "Answer with only 'yes', 'no', or 'maybe'.\n\n"
            "The correct answer is"
        )


def build_prompt(
    sample: QASample, condition: Condition, context: Optional[str] = None
) -> str:
    if sample.source == "medqa":
        return build_prompt_medqa(sample, condition, context)
    return build_prompt_pubmedqa(sample, condition, context)


# ── Answer extractor ────────────────────────────────────────────────────────
def extract_answer(text: str, source: str) -> Optional[str]:
    """Pull out the structured answer from raw model output."""
    text = text.strip()
    if source == "medqa":
        m = re.search(r"Answer:\s*([A-D])", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m = re.match(r"^\s*([A-D])\b", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        return None
    if source == "pubmedqa":
        lower = text.lower()
        for label in ["yes", "no", "maybe"]:
            if lower.startswith(label):
                return label
        m = re.search(r"\b(yes|no|maybe)\b", lower)
        return m.group(1) if m else None
    return None


# ── Model runner ────────────────────────────────────────────────────────────
class ModelRunner:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        print(f"Loading {model_name}...")
        print(f"CUDA available: {torch.cuda.is_available()}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # BioGPT and Mistral have no pad token by default — use EOS
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # left-padding for decoder-only models during batched inference
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.model.eval()

    def run(
        self,
        samples: list[QASample],
        condition: Condition,
        contexts: Optional[list[Optional[str]]] = None,
        batch_size: int = 8,
    ) -> list[InferenceResult]:
        results = []
        contexts = contexts or [None] * len(samples)

        for batch_start in range(0, len(samples), batch_size):
            batch_samples  = samples[batch_start: batch_start + batch_size]
            batch_contexts = contexts[batch_start: batch_start + batch_size]

            prompts = [
                build_prompt(s, condition, ctx)
                for s, ctx in zip(batch_samples, batch_contexts)
            ]

            # warn on long prompts before tokenizing
            for s, prompt in zip(batch_samples, prompts):
                prompt_tokens = len(self.tokenizer(prompt)["input_ids"])
                if prompt_tokens > 768:
                    print(f"WARNING: prompt for {s.id} is {prompt_tokens} tokens — may truncate")

            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,   # letter or yes/no/maybe — 64 is plenty
                    do_sample=False,     # greedy, reproducible
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # decode only newly generated tokens — strip the prompt prefix
            prompt_len = inputs["input_ids"].shape[1]
            for s, output_ids, prompt in zip(batch_samples, outputs, prompts):
                generated = self.tokenizer.decode(
                    output_ids[prompt_len:],
                    skip_special_tokens=True,
                ).strip()
                results.append(InferenceResult(
                    sample_id=s.id,
                    model_name=self.model_name,
                    condition=condition,
                    raw_output=generated,
                    extracted_answer=extract_answer(generated, s.source),
                    prompt_used=prompt,
                ))

            print(f"  [{batch_start + len(batch_samples)}/{len(samples)}] batches done")

        return results

    def save(self, results: list[InferenceResult], path: str):
        with open(path, "w") as f:
            json.dump([vars(r) for r in results], f, indent=2)
        print(f"Saved {len(results)} results → {path}")


# ── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from load_dataset import load_pubmedqa, stratified_sample
    samples = stratified_sample(load_pubmedqa("train"), n=5)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    runner = ModelRunner("microsoft/biogpt", device=device)
    results = runner.run(samples, condition="zero_shot")
    for r in results:
        print(f"{r.sample_id}: extracted={r.extracted_answer!r}")
        print(f"  raw: {r.raw_output[:120]}")
        print()
    runner.save(results, "outputs/biogpt_zero_shot.json")