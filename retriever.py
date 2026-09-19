"""
retriever.py
Loads text files from the knowledge base directories and performs
TF-IDF similarity search to return the most relevant passages.

v3: No logic changes from v2. Source tracking, top-k results, and
CONFIDENCE_THRESHOLD already in place. Comments tidied.
"""

from __future__ import annotations

import os
import math
import re
from collections import defaultdict
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KB_DIRS = [
    "data/hackerrank",
    "data/visa",
    "data/claude",
]

CONFIDENCE_THRESHOLD    = 0.08   # minimum score for decision-gating
JUSTIFICATION_THRESHOLD = 0.05   # minimum score to mention in justification


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    passage: str
    score:   float
    source:  str    # relative filepath, e.g. "data/visa/faq.txt"
    rank:    int    # 1 = best match


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _chunk_text(text: str, chunk_size: int = 5) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks: list[str] = []
    step = max(1, chunk_size // 2)
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# KnowledgeBase
# ---------------------------------------------------------------------------

class KnowledgeBase:
    def __init__(self, base_path: str = "."):
        self.passages:        list[str]  = []
        self.passage_sources: list[str]  = []
        self.tf_vectors:      list[dict] = []
        self.idf:             dict[str, float] = {}
        self._load(base_path)
        self._build_idf()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, base_path: str) -> None:
        for rel_dir in KB_DIRS:
            dir_path = os.path.join(base_path, rel_dir)
            if not os.path.isdir(dir_path):
                continue
            for fname in sorted(os.listdir(dir_path)):
                fpath = os.path.join(dir_path, fname)
                if not os.path.isfile(fpath):
                    continue
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    rel_source = os.path.join(rel_dir, fname)
                    for chunk in _chunk_text(content):
                        self.passages.append(chunk)
                        self.passage_sources.append(rel_source)
                        self.tf_vectors.append(self._tf(chunk))
                except OSError:
                    continue

    def _tf(self, text: str) -> dict[str, float]:
        tokens = _tokenize(text)
        if not tokens:
            return {}
        counts: dict[str, int] = defaultdict(int)
        for t in tokens:
            counts[t] += 1
        total = len(tokens)
        return {t: c / total for t, c in counts.items()}

    # ------------------------------------------------------------------
    # IDF
    # ------------------------------------------------------------------

    def _build_idf(self) -> None:
        n = len(self.tf_vectors)
        if n == 0:
            return
        df: dict[str, int] = defaultdict(int)
        for vec in self.tf_vectors:
            for term in vec:
                df[term] += 1
        self.idf = {
            term: math.log((n + 1) / (freq + 1)) + 1
            for term, freq in df.items()
        }

    def _tfidf(self, tf_vec: dict[str, float]) -> dict[str, float]:
        return {t: tf * self.idf.get(t, 0.0) for t, tf in tf_vec.items()}

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        common = set(a) & set(b)
        if not common:
            return 0.0
        dot   = sum(a[t] * b[t] for t in common)
        mag_a = math.sqrt(sum(v * v for v in a.values()))
        mag_b = math.sqrt(sum(v * v for v in b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_top_k(self, query: str, k: int = 3) -> list[SearchResult]:
        """Return the top-k most relevant passages, sorted by descending score."""
        if not self.passages:
            return []

        query_vec = self._tfidf(self._tf(query))
        scored: list[tuple[float, int]] = []

        for idx, tf_vec in enumerate(self.tf_vectors):
            score = self._cosine(query_vec, self._tfidf(tf_vec))
            scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            SearchResult(
                passage=self.passages[idx],
                score=round(score, 4),
                source=self.passage_sources[idx],
                rank=rank + 1,
            )
            for rank, (score, idx) in enumerate(scored[: min(k, len(scored))])
        ]

    def search(self, query: str) -> tuple[str | None, float]:
        """Backward-compatible single-result search."""
        results = self.search_top_k(query, k=1)
        if not results:
            return None, 0.0
        return results[0].passage, results[0].score

    def is_confident(self, score: float) -> bool:
        return score >= CONFIDENCE_THRESHOLD

    def meets_justification_bar(self, score: float) -> bool:
        return score >= JUSTIFICATION_THRESHOLD