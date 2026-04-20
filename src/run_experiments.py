"""
run_experiments.py
Single entry point to execute all 10 experiment runs
from the matrix in sequence. Each team member can comment
out the runs they're not responsible for and run this script.

Usage:
  python run_experiments.py --runs 1 2 3          # run specific runs
  python run_experiments.py --runs all             # run everything
  python run_experiments.py --runs 7 --n 50        # quick smoke test (n=50)
"""
import os
os.environ["HF_HOME"] = f"/projectnb/cs505am/projects/empirical_hallucination/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]

import argparse, json
from pathlib import Path

from load_dataset import load_medqa, load_pubmedqa, stratified_sample
from inference    import ModelRunner
from rag          import BM25Retriever
from evaluate     import Evaluator

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# ── Model checkpoint paths ────────────────────────────────────────────────────
MODELS = {
    # "mistralinstruct":    "mistralai/Mistral-7B-Instruct-v0.3",
    "llama3":   "meta-llama/Meta-Llama-3-8B-Instruct", 
    "biomistral":"BioMistral/BioMistral-7B",
    "biogpt":    "microsoft/biogpt",
    "biogpt_sft":"checkpoints/biogpt-medqa/best",       # set after finetune.py runs
}

# ── Shared data (loaded once) ─────────────────────────────────────────────────
def load_data(n: int, seed: int = 42):
    medqa_samples    = stratified_sample(load_medqa("test"),   n, seed)
    pubmedqa_samples = stratified_sample(load_pubmedqa("train"), n, seed)
    medqa_map        = {s.id: s for s in medqa_samples}
    pubmedqa_map     = {s.id: s for s in pubmedqa_samples}
    retriever        = BM25Retriever.from_pubmedqa()
    return medqa_samples, pubmedqa_samples, medqa_map, pubmedqa_map, retriever


# ── Single run helper ─────────────────────────────────────────────────────────
def run_one(
    run_id: int,
    model_key: str,
    condition: str,
    samples,
    sample_map,
    retriever=None,
    run_minicheck: bool = False,
    rag_top_k: int = 1,   
):
    if condition == "rag":
        tag = f"run{run_id:02d}_{model_key}_{condition}_top{rag_top_k}"
    else:
        tag = f"run{run_id:02d}_{model_key}_{condition}"
    inf_path  = OUT / f"{tag}_inference.json"
    eval_path = OUT / f"eval_{tag}.json"

    if eval_path.exists():
        print(f"[run {run_id}] Already done — skipping ({eval_path.name})")
        return

    print(f"\n{'='*60}")
    print(f"[run {run_id}] model={model_key}  condition={condition}  rag_top_k={rag_top_k if retriever and 'rag' in condition else 'N/A'}")
    print(f"{'='*60}")

    # ── inference ─────────────────────────────────────────────────────────
    contexts      = None
    top_k_used    = None
    if retriever and "rag" in condition:
        print(f"  Retrieving contexts (BM25, k={rag_top_k})...")
        contexts, top_k_used = retriever.retrieve_batch(samples, k=rag_top_k)

    runner  = ModelRunner(MODELS[model_key])
    results = runner.run(samples, condition=condition, contexts=contexts, rag_top_k=top_k_used)
    runner.save(results, str(inf_path))

    # ── evaluation ────────────────────────────────────────────────────────
    ev      = Evaluator(sample_map, run_bertscore=True, run_minicheck=run_minicheck)
    records = ev.evaluate(results)
    ev.save(records, str(eval_path))
    print(f"  Done. eval → {eval_path.name}")


# ── Experiment matrix ─────────────────────────────────────────────────────────
def build_matrix(medqa, pubmedqa, medqa_map, pubmedqa_map, retriever, rag_top_k):
    # (run_id, model_key, condition, samples, sample_map, retriever, minicheck, rag_top_k)
    return [
        ( 1, "llama3",          "zero_shot", medqa,    medqa_map,    None,      False, None),
        ( 2, "llama3",          "cot",       medqa,    medqa_map,    None,      False, None),
        ( 3, "llama3",          "rag",       pubmedqa, pubmedqa_map, retriever, False, rag_top_k),
        ( 4, "biomistral",      "zero_shot", medqa,    medqa_map,    None,      False, None),
        ( 5, "biomistral",      "cot",       medqa,    medqa_map,    None,      False, None),
        ( 6, "biomistral",      "rag",       pubmedqa, pubmedqa_map, retriever, False, rag_top_k),
        ( 7, "biogpt",          "zero_shot", medqa,    medqa_map,    None,      False, None),
        ( 8, "biogpt",          "rag",       pubmedqa, pubmedqa_map, retriever, False, rag_top_k),
        ( 9, "biogpt_sft",      "sft",       medqa,    medqa_map,    None,      False, None),
        (10, "biogpt_sft",      "sft_rag",   pubmedqa, pubmedqa_map, retriever, False, rag_top_k),
    ]

# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs", nargs="+", default=["all"],
        help="Run IDs to execute (e.g. 1 2 3) or 'all'"
    )
    parser.add_argument(
        "--n", type=int, default=200,
        help="Samples per dataset (default 200; use 50 for a quick smoke test)"
    )
    parser.add_argument(
        "--rag_top_k", type=int, default=1,       # new CLI arg
        help="Number of abstracts to retrieve for RAG conditions (default 1)"
    )
    args = parser.parse_args()

    target_runs = (
        None if args.runs == ["all"]
        else {int(r) for r in args.runs}
    )

    print(f"Loading datasets (n={args.n} per dataset)...")
    medqa, pubmedqa, medqa_map, pubmedqa_map, retriever = load_data(args.n)

    matrix = build_matrix(medqa, pubmedqa, medqa_map, pubmedqa_map, retriever, args.rag_top_k)

    for row in matrix:
        run_id = row[0]
        if target_runs and run_id not in target_runs:
            continue
        # skip run 9 & 10 if fine-tuned checkpoint doesn't exist yet
        if run_id in (9, 10):
            ckpt = Path(MODELS["biogpt_sft"])
            if not ckpt.exists():
                print(f"[run {run_id}] Skipping — fine-tuned checkpoint not found.")
                print(f"  Run finetune.py first, then re-run with --runs {run_id}")
                continue
        run_one(*row)

    print("\nAll requested runs complete. Run analyze.py to generate results.")
