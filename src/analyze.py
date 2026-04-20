"""
analyze.py
Loads all eval records, produces:
  1. results_summary.csv   — one row per (model, condition, dataset)
  2. Hallucination rate bar chart
  3. Accuracy by condition grouped bar chart
  4. Error taxonomy breakdown
  5. Interaction delta table (RQ4)

Install: pip install pandas matplotlib seaborn scipy
"""

from __future__ import annotations
import json, glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from evaluate import EvalRecord, aggregate

OUTPUT_DIR = Path("outputs/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load all eval JSON files ─────────────────────────────────────────────────
def load_all_records(eval_dir: str = "outputs") -> pd.DataFrame:
    """
    Glob all eval_*.json files and merge into a single DataFrame.
    Each file corresponds to one (model, condition) run.
    """
    rows = []
    for path in glob.glob(f"{eval_dir}/eval_*.json"):
        with open(path) as f:
            records = [EvalRecord(**r) for r in json.load(f)]
        rows.extend([vars(r) for r in records])
    return pd.DataFrame(rows)


# ── Summary table ────────────────────────────────────────────────────────────
def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics grouped by (model, condition, source)."""
    groups = df.groupby(["model_name", "condition", "source", "rag_top_k"])
    rows = []
    for (model, cond, src), grp in groups:
        records = [EvalRecord(**r) for r in grp.to_dict("records")]
        row = aggregate(records)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "results_summary.csv", index=False)
    print("Saved results_summary.csv")
    return summary


# ── Plot helpers ─────────────────────────────────────────────────────────────
CONDITION_ORDER = ["zero_shot", "cot", "rag", "sft", "sft_rag"]
PALETTE = "Set2"

def plot_hallucination_rates(summary: pd.DataFrame):
    """
    Grouped bar chart: hallucination rate per condition, one bar per model.
    Separate subplots for MedQA vs PubMedQA.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, src in zip(axes, ["medqa", "pubmedqa"]):
        sub = summary[summary["source"] == src].copy()
        sub["condition"] = pd.Categorical(
            sub["condition"], categories=CONDITION_ORDER, ordered=True
        )
        sub = sub.sort_values("condition")
        sns.barplot(
            data=sub, x="condition", y="hallucination_rate",
            hue="model", palette=PALETTE, ax=ax
        )
        ax.set_title(src.upper(), fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("Hallucination rate" if ax == axes[0] else "")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title="Model", fontsize=8)
    fig.suptitle("Hallucination rate by condition and model", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hallucination_rates.pdf", dpi=150)
    plt.close()
    print("Saved hallucination_rates.pdf")


def plot_accuracy(summary: pd.DataFrame):
    """Accuracy on MedQA only — grouped bar by model × condition."""
    sub = summary[summary["source"] == "medqa"].dropna(subset=["accuracy"])
    sub["condition"] = pd.Categorical(
        sub["condition"], categories=CONDITION_ORDER, ordered=True
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=sub, x="condition", y="accuracy",
        hue="model", palette=PALETTE, ax=ax
    )
    ax.set_ylim(0, 1)
    ax.set_title("MedQA accuracy by condition and model")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "accuracy_by_condition.pdf", dpi=150)
    plt.close()
    print("Saved accuracy_by_condition.pdf")


def plot_error_taxonomy(df: pd.DataFrame):
    """
    Stacked bar of manual error taxonomy labels, if the column exists.
    Only run this after the annotation sprint (Week 5-6).
    Column expected: 'error_type' with values from the qualitative taxonomy.
    """
    if "error_type" not in df.columns:
        print("Skipping error taxonomy — 'error_type' column not present yet.")
        return
    hallucinated = df[df["is_hallucination"] == True]
    counts = (
        hallucinated.groupby(["condition", "error_type"])
        .size().unstack(fill_value=0)
    )
    counts.plot(kind="bar", stacked=True, colormap="Set3", figsize=(10, 5))
    plt.title("Hallucination error types by condition")
    plt.xlabel("")
    plt.ylabel("Count")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "error_taxonomy.pdf", dpi=150)
    plt.close()
    print("Saved error_taxonomy.pdf")


# ── RQ4: Interaction delta ───────────────────────────────────────────────────
def interaction_delta_table(summary: pd.DataFrame) -> pd.DataFrame:
    """
    For BioGPT only, compute:
      delta_rag     = hallucination(zero_shot) - hallucination(rag)
      delta_sft     = hallucination(zero_shot) - hallucination(sft)
      delta_sft_rag = hallucination(zero_shot) - hallucination(sft_rag)
      interaction   = delta_sft_rag - (delta_rag + delta_sft)
                      positive → synergistic; negative → redundant
    """
    biogpt = summary[
        summary["model"].str.contains("biogpt", case=False)
    ].set_index(["condition", "source"])

    rows = []
    for src in ["medqa", "pubmedqa"]:
        try:
            h = lambda cond: biogpt.loc[(cond, src), "hallucination_rate"]
            d_rag     = h("zero_shot") - h("rag")
            d_sft     = h("zero_shot") - h("sft")
            d_sft_rag = h("zero_shot") - h("sft_rag")
            interaction = d_sft_rag - (d_rag + d_sft)
            rows.append({
                "source": src,
                "Δ RAG": round(d_rag, 4),
                "Δ SFT": round(d_sft, 4),
                "Δ SFT+RAG": round(d_sft_rag, 4),
                "interaction": round(interaction, 4),
                "verdict": "synergistic" if interaction > 0 else "redundant",
            })
        except KeyError:
            pass

    delta_df = pd.DataFrame(rows)
    delta_df.to_csv(OUTPUT_DIR / "interaction_delta.csv", index=False)
    print("Saved interaction_delta.csv")
    return delta_df


# ── Statistical significance ─────────────────────────────────────────────────
def significance_tests(df: pd.DataFrame):
    """
    McNemar's test for paired conditions on the same samples.
    Prints a table of p-values for each (model, condition_A vs condition_B) pair.
    """
    pairs = [
        ("zero_shot", "rag"),
        ("zero_shot", "sft"),
        ("sft", "sft_rag"),
    ]
    print("\n── Significance tests (McNemar) ──")
    for model, grp in df.groupby("model_name"):
        print(f"\n{model}")
        for cond_a, cond_b in pairs:
            a = grp[grp["condition"] == cond_a].set_index("sample_id")
            b = grp[grp["condition"] == cond_b].set_index("sample_id")
            shared = a.index.intersection(b.index)
            if len(shared) < 10:
                continue
            a_wrong = a.loc[shared, "is_hallucination"].astype(int)
            b_wrong = b.loc[shared, "is_hallucination"].astype(int)
            # contingency table: [[both right, a right b wrong], ...]
            n01 = ((a_wrong == 0) & (b_wrong == 1)).sum()
            n10 = ((a_wrong == 1) & (b_wrong == 0)).sum()
            if n01 + n10 == 0:
                continue
            chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
            p = stats.chi2.sf(chi2, df=1)
            sig = "*" if p < 0.05 else "ns"
            print(f"  {cond_a} vs {cond_b}: p={p:.4f} {sig}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_all_records()
    if df.empty:
        print("No eval records found. Run evaluate.py first.")
    else:
        summary = build_summary(df)
        print(summary.to_string())
        plot_hallucination_rates(summary)
        plot_accuracy(summary)
        plot_error_taxonomy(df)
        delta = interaction_delta_table(summary)
        print(delta.to_string())
        significance_tests(df)