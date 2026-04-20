"""
rag.py
BM25-based retriever over PubMedQA abstracts.
Returns one context string per query sample.
Dense retrieval (MedCPT) is a stretch upgrade — same interface.

Install: pip install rank-bm25
"""

from __future__ import annotations
import re
from typing import Optional
from rank_bm25 import BM25Okapi
from load_dataset import QASample, load_pubmedqa

TOP_K = 2   # number of abstracts to retrieve per query


def _tokenize(text: str) -> list[str]:
    """Lowercase + whitespace tokenisation for BM25."""
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """
    Indexes the PubMedQA abstract corpus once, then retrieves
    the top-k most relevant abstracts for any query string.

    Usage:
        retriever = BM25Retriever.from_pubmedqa()
        contexts = retriever.retrieve_batch(query_samples, k=3)
        # pass contexts into ModelRunner.run(..., contexts=contexts)
    """

    def __init__(self, corpus: list[QASample]):
        self._corpus = corpus
        self._abstracts = [s.context or "" for s in corpus]
        tokenized = [_tokenize(a) for a in self._abstracts]
        self._bm25 = BM25Okapi(tokenized)
        print(f"BM25 index built over {len(corpus)} abstracts.")

    @classmethod
    def from_pubmedqa(cls) -> "BM25Retriever":
        """Convenience: build index directly from the PubMedQA corpus."""
        corpus = load_pubmedqa(split="train")
        return cls(corpus)

    def retrieve(self, query: str, k: int = TOP_K) -> str:
        """
        Retrieve top-k abstracts for a single query.
        Returns a single concatenated context string ready to
        drop into a prompt template.
        """
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        parts = [
            f"[Abstract {rank+1}] {self._abstracts[i]}"
            for rank, i in enumerate(top_indices)
            if self._abstracts[i]
        ]
        return "\n\n".join(parts)

    def retrieve_batch(
        self,
        samples: list[QASample],
        k: int = TOP_K,
    ) -> list[Optional[str]]:
        """
        Retrieve contexts for a list of samples.
        Returns a list aligned 1-to-1 with samples —
        pass directly as `contexts` arg in ModelRunner.run().
        """
        contexts = [self.retrieve(s.question, k=k) for s in samples]
        return contexts, k


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from load_dataset import load_medqa, stratified_sample
    retriever = BM25Retriever.from_pubmedqa()
    samples = stratified_sample(load_medqa("test"), n=3)
    contexts = retriever.retrieve_batch(samples)
    for s, ctx in zip(samples, contexts):
        print(f"Q: {s.question[:80]}...")
        print(f"Retrieved ({len(ctx)} chars):\n{ctx[:300]}...\n")