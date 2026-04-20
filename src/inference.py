"""
inference.py
Unified model wrapper for zero-shot, CoT, and RAG conditions.
Produces raw text output + extracted answer letter for evaluation.
"""

import re
import json
from dataclasses import dataclass, field
from typing import Optional, Literal
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
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
    is_hallucination: Optional[bool] = field(default=None) # filled in by evaluate.py = field(default=None)
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
    # only use context if explicitly passed (RAG condition)
    # never fall back to sample.context here — that's for evaluate.py only
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
        # Look for explicit "Answer: X" first (CoT), then bare letter
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
    """
    Thin wrapper around a HuggingFace causal-LM pipeline.
    Supports any checkpoint loadable via AutoModelForCausalLM.

    Usage:
        runner = ModelRunner("microsoft/biogpt")
        results = runner.run(samples, condition="zero_shot")
    """
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # token = True,
            torch_dtype=torch.float16,
            # device_map=device,
        ).to(device)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            # device=0 if device == "cuda" else -1,
            max_new_tokens=256,
            do_sample=False,         # greedy for reproducibility
            temperature=None,
            top_p=None,
            return_full_text=False,   # <-- add this
        )

    def run(
        self,
        samples: list[QASample],
        condition: Condition,
        contexts: Optional[list[Optional[str]]] = None,
    ) -> list[InferenceResult]:
        """
        Run inference on a list of samples.
        contexts: optional list of retrieved contexts (one per sample, for RAG).
        """
        results = []
        contexts = contexts or [None] * len(samples)

        for sample, ctx in zip(samples, contexts):
            prompt = build_prompt(sample, condition, ctx)
            # guard: warn if prompt is dangerously long
            prompt_tokens = len(self.tokenizer(prompt)["input_ids"])
            if prompt_tokens > 768:  # leaves ~256 tokens for generation
                print(f"WARNING: prompt for {sample.id} is {prompt_tokens} tokens — may truncate output")
    
            output = self.pipe(prompt)[0]["generated_text"]
            # strip the prompt prefix that the pipeline echoes back
            generated = output.strip() # generated = output[len(prompt):].strip()
            results.append(InferenceResult(
                sample_id=sample.id,
                model_name=self.model_name,
                condition=condition,
                raw_output=generated,
                extracted_answer=extract_answer(generated, sample.source),
                prompt_used=prompt,
            ))
        return results

    def save(self, results: list[InferenceResult], path: str):
        with open(path, "w") as f:
            json.dump([vars(r) for r in results], f, indent=2)
        print(f"Saved {len(results)} results → {path}")


# ── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from load_dataset import load_pubmedqa, stratified_sample
    samples = stratified_sample(load_pubmedqa("test"), n=5)
    device = torch.device("cuda") if (torch.cuda.is_available()) else torch.device("cpu")

    runner = ModelRunner("microsoft/biogpt", device = device)
    
    results = runner.run(samples, condition="zero_shot")
    for r in results:
        print(f"{r.sample_id}: extracted={r.extracted_answer!r}")
        print(f"  raw: {r.raw_output[:120]}")
        print()

    runner.save(results, "outputs/biogpt_zero_shot.json")