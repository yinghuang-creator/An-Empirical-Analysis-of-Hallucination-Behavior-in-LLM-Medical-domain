"""
load_dataset.py
Loads, normalises, and samples MedQA and PubMedQA into a
shared schema for downstream inference + evaluation.
"""

import random
from datasets import load_dataset
from dataclasses import dataclass
from typing import Optional

SEED = 42

# ── Shared schema ──────────────────────────────────────────────────────────
@dataclass
class QASample:
    id: str
    source: str                       # 'medqa' | 'pubmedqa'
    question: str
    context: Optional[str]            # abstract for PubMedQA; None for MedQA
    choices: Optional[dict]           # {'A': ..., 'B': ..., ...} for MedQA
    gold_answer: str                  # letter (A-D) for MedQA; yes/no/maybe for PubMedQA
    gold_long_answer: Optional[str]   # full answer text if available


# ── Internal parquet loader ─────────────────────────────────────────────────
def _load_parquet(repo: str, subset_dir: str, split: str):
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split '{split}'")
    url = (
        f"hf://datasets/{repo}@refs/convert/parquet/"
        f"{subset_dir}/{split}/0000.parquet"
    )
    return load_dataset("parquet", data_files={split: url}, split=split)


# ── MedQA ───────────────────────────────────────────────────────────────────
def load_medqa(split: str = "test") -> list[QASample]:
    """
    Raw fields: question, answer, options (dict A-D), meta_info, answer_idx
    We default to 'test' split — it's the held-out eval set.
    """
    ds = _load_parquet("bigbio/med_qa", "med_qa_en_4options_source", split)
    samples = []
    for i, row in enumerate(ds):
        samples.append(QASample(
            id=f"medqa-{split}-{i}",
            source="medqa",
            question=row["question"],
            context=None,
            choices=row["options"],      # already a dict
            gold_answer=row["answer_idx"],  # letter e.g. 'A'
            gold_long_answer=row["answer"],
        ))
    return samples


# ── PubMedQA ────────────────────────────────────────────────────────────────
def load_pubmedqa(split: str = "train") -> list[QASample]:
    """
    Raw fields: pubid, question, context (dict with 'contexts' list + 'labels'),
    long_answer, final_decision (yes/no/maybe).
    PubMedQA only has a labelled train split of ~1000; use it for all conditions.
    """
    ds = _load_parquet(
        "bigbio/pubmed_qa", "pubmed_qa_labeled_fold0_source", split
    )
    samples = []
    for i,row in enumerate(ds):
        abstract = " ".join(row["CONTEXTS"])
        samples.append(QASample(
            id=f"pubmedqa-{i}",
            source="pubmedqa",
            question=row["QUESTION"],
            context=abstract,
            choices=None,
            gold_answer=row["final_decision"],   # 'yes' | 'no' | 'maybe'
            gold_long_answer=row["LONG_ANSWER"],
        ))
    return samples


# ── Sampling ────────────────────────────────────────────────────────────────
def stratified_sample(
    samples: list[QASample],
    n: int,
    seed: int = SEED,
) -> list[QASample]:
    """
    Sample n items with balanced gold_answer distribution.
    For MedQA this balances across A/B/C/D.
    For PubMedQA this balances across yes/no/maybe.
    Falls back to plain random sample if a class has too few items.
    """
    rng = random.Random(seed)
    by_class: dict[str, list[QASample]] = {}
    for s in samples:
        by_class.setdefault(s.gold_answer, []).append(s)

    classes = sorted(by_class)
    per_class = n // len(classes)
    result = []
    for cls in classes:
        pool = by_class[cls]
        rng.shuffle(pool)
        result.extend(pool[:per_class])

    # top-up with random remainder if needed
    ids = [res.id for res in result]
    remaining = [s for s in samples if s.id not in set(ids)]
    rng.shuffle(remaining)
    result.extend(remaining[: n - len(result)])
    rng.shuffle(result)
    return result[:n]


# ── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    medqa = load_medqa(split="test")
    sample = stratified_sample(medqa, n=200)
    print(f"MedQA test set: {len(medqa)} items → sampled {len(sample)}")
    print("Answer distribution:", {
        k: sum(1 for s in sample if s.gold_answer == k)
        for k in sorted({s.gold_answer for s in sample})
    })
    print("\nSample item:")
    print(sample[0])

    pubmedqa = load_pubmedqa()
    psample = stratified_sample(pubmedqa, n=200)
    print(f"\nPubMedQA: {len(pubmedqa)} items → sampled {len(psample)}")
    print("Decision distribution:", {
        k: sum(1 for s in psample if s.gold_answer == k)
        for k in ["yes", "no", "maybe"]
    })