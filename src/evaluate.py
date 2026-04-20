"""
evaluate.py
Scores InferenceResult objects against gold labels.

Metrics:
  - Accuracy / hallucination rate  (all conditions, MedQA)
  - MiniCheck faithfulness score    (RAG conditions, PubMedQA)
  - BERTScore F1                    (secondary, all conditions)

Install: pip install bert-score minicheck
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Optional
from bert_score import score as bert_score_fn
from load_dataset import QASample
from inference import InferenceResult

# lazy import — MiniCheck is heavyweight, only load when needed
_minicheck = None

def _get_minicheck():
    global _minicheck
    if _minicheck is None:
        from minicheck.minicheck import MiniCheck
        _minicheck = MiniCheck(model_name="roberta-large")
    return _minicheck


# ── Per-sample eval record ───────────────────────────────────────────────────
@dataclass
class EvalRecord:
    sample_id: str
    source: str                        # 'medqa' | 'pubmedqa'
    model_name: str
    condition: str
    gold_answer: str
    extracted_answer: Optional[str]
    is_correct: Optional[bool]         # None if extraction failed
    is_hallucination: bool             # True when incorrect or unfaithful
    parse_failed: bool                 # True when extracted_answer is None
    rag_top_k: Optional[int] = field(default=None)
    bertscore_f1: Optional[float] = field(default=None)
    minicheck_label: Optional[str] = field(default=None)   # 'faithful'|'unfaithful'
    minicheck_prob: Optional[float] = field(default=None)


# ── Core evaluator ───────────────────────────────────────────────────────────
class Evaluator:
    """
    Evaluate a list of InferenceResults against their source QASamples.

    sample_map: dict mapping sample_id → QASample (build once, reuse).
    run_bertscore: set False to skip (slow on CPU).
    run_minicheck: set True only for RAG/PubMedQA conditions.
    """
    def __init__(
        self,
        sample_map: dict[str, QASample],
        run_bertscore: bool = True,
        run_minicheck: bool = False,
    ):
        self.sample_map = sample_map
        self.run_bertscore = run_bertscore
        self.run_minicheck = run_minicheck

    def evaluate(self, results: list[InferenceResult]) -> list[EvalRecord]:
        records = []

        # ── BERTScore (batch for efficiency) ────────────────────────────────
        bertscore_map: dict[str, float] = {}
        if self.run_bertscore:
            preds = [r.raw_output for r in results]
            refs  = [
                self.sample_map[r.sample_id].gold_long_answer or ""
                for r in results
            ]
            _, _, F1 = bert_score_fn(preds, refs, lang="en", verbose=False)
            bertscore_map = {
                r.sample_id: float(f1)
                for r, f1 in zip(results, F1)
            }

        # ── MiniCheck (batch for RAG/PubMedQA) ─────────────────────────────
        minicheck_map: dict[str, tuple[str, float]] = {}
        if self.run_minicheck:
            mc = _get_minicheck()
            rag_results = [
                r for r in results
                if self.sample_map[r.sample_id].context
            ]
            if rag_results:
                docs   = [self.sample_map[r.sample_id].context for r in rag_results]
                claims = [r.raw_output for r in rag_results]
                preds, probs, _, _ = mc.score(docs=docs, claims=claims)
                for r, pred, prob in zip(rag_results, preds, probs):
                    label = "faithful" if pred == 1 else "unfaithful"
                    minicheck_map[r.sample_id] = (label, float(prob))

        # ── Per-sample record ───────────────────────────────────────────────
        for r in results:
            sample = self.sample_map[r.sample_id]
            parse_failed = r.extracted_answer is None

            if parse_failed:
                is_correct = None
                is_hallucination = True   # treat unparseable output as wrong
            else:
                is_correct = (
                    r.extracted_answer.lower() == sample.gold_answer.lower()
                )
                is_hallucination = not is_correct

            mc_label, mc_prob = minicheck_map.get(r.sample_id, (None, None))
            # for RAG conditions, also flag minicheck unfaithful as hallucination
            if mc_label == "unfaithful":
                is_hallucination = True

            records.append(EvalRecord(
                sample_id=r.sample_id,
                source=sample.source,
                model_name=r.model_name,
                condition=r.condition,
                gold_answer=sample.gold_answer,
                extracted_answer=r.extracted_answer,
                is_correct=is_correct,
                is_hallucination=is_hallucination,
                rag_top_k=r.rag_top_k, 
                parse_failed=parse_failed,
                bertscore_f1=bertscore_map.get(r.sample_id),
                minicheck_label=mc_label,
                minicheck_prob=mc_prob,
            ))
        return records

    def save(self, records: list[EvalRecord], path: str):
        with open(path, "w") as f:
            json.dump([vars(r) for r in records], f, indent=2)
        print(f"Saved {len(records)} eval records → {path}")


# ── Aggregate metrics helper ─────────────────────────────────────────────────
def aggregate(records: list[EvalRecord]) -> dict:
    """
    Compute summary stats for a batch of EvalRecords.
    Returns a dict suitable for adding as a row in a results DataFrame.
    """
    total = len(records)
    if total == 0:
        return {}
    parseable = [r for r in records if not r.parse_failed]
    hallucinated = [r for r in records if r.is_hallucination]
    faithful = [r for r in records if r.minicheck_label == "faithful"]
    mc_total = [r for r in records if r.minicheck_label is not None]
    bs_vals = [r.bertscore_f1 for r in records if r.bertscore_f1 is not None]

    return {
        "model":             records[0].model_name,
        "condition":         records[0].condition,
        "source":            records[0].source,
        "rag_top_k":          records[0].rag_top_k,
        "n":                  total,
        "parse_fail_rate":   round(1 - len(parseable) / total, 4),
        "accuracy":          round(
            sum(1 for r in parseable if r.is_correct) / len(parseable), 4
        ) if parseable else None,
        "hallucination_rate": round(len(hallucinated) / total, 4),
        "faithfulness_rate":  round(len(faithful) / len(mc_total), 4)
                              if mc_total else None,
        "bertscore_f1_mean": round(sum(bs_vals) / len(bs_vals), 4)
                              if bs_vals else None,
    }


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    from load_dataset import load_pubmedqa, stratified_sample

    samples = stratified_sample(load_pubmedqa("test"), n=5)
    sample_map = {s.id: s for s in samples}

    # load previously saved inference results
    with open("outputs/biogpt_zero_shot.json") as f:
        raw = json.load(f)
    from inference import InferenceResult
    results = [InferenceResult(**r) for r in raw]

    ev = Evaluator(sample_map, run_bertscore=True, run_minicheck=False)
    records = ev.evaluate(results)
    ev.save(records, "outputs/eval_biogpt_zero_shot.json")
    print(aggregate(records))