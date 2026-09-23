"""BM25 full-text ranking via hand-built inverted index.

Standard BM25 formula: score = sum over query terms of
    idf(t) * tf(t,D) * (k1+1) / (tf(t,D) + k1 * (1 - b + b * |D| / avgdl))

where:
- idf(t) = ln((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
- tf(t,D) = count of term t in document D
- |D| = length of document D (in tokens)
- avgdl = average document length across the corpus
"""

import json
import re
from pathlib import Path

from engine import config


class InvertedIndex:
    def __init__(self, postings: dict, doc_lengths: list[int], N: int):
        """
        Args:
            postings: {term: [(doc_idx, tf), ...], ...}
            doc_lengths: [num_tokens_in_doc_0, ...]
            N: total number of documents
        """
        self.postings = postings
        self.doc_lengths = doc_lengths
        self.N = N
        self.avgdl = sum(doc_lengths) / N if N > 0 else 1.0

    @classmethod
    def build(cls, token_lists: list[list[str]]) -> "InvertedIndex":
        """Build index from per-document token lists.

        Args:
            token_lists: list[i] = tokens for document i

        Returns: InvertedIndex ready for scoring"""
        postings = {}
        doc_lengths = []

        for doc_idx, tokens in enumerate(token_lists):
            doc_lengths.append(len(tokens))
            token_counts = {}
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1

            for token, tf in token_counts.items():
                if token not in postings:
                    postings[token] = []
                postings[token].append((doc_idx, tf))

        return cls(postings, doc_lengths, len(token_lists))

    def bm25_scores(self, query_tokens: list[str], candidate_indices: set[int]) -> dict:
        """Score candidates via BM25 for the given query tokens.

        Args:
            query_tokens: list of query tokens (pre-tokenized)
            candidate_indices: set of doc indices to score

        Returns: {doc_idx: score}"""
        scores = {idx: 0.0 for idx in candidate_indices}

        for token in query_tokens:
            if token not in self.postings:
                continue

            df = len(self.postings[token])
            idf = (
                1.0
                + __import__("math").log(
                    (self.N - df + 0.5) / (df + 0.5)
                )
            )

            for doc_idx, tf in self.postings[token]:
                if doc_idx not in candidate_indices:
                    continue

                doc_len = self.doc_lengths[doc_idx]
                norm_factor = 1 - config.BM25_B + config.BM25_B * (
                    doc_len / self.avgdl
                )
                bm25_term = idf * tf * (config.BM25_K1 + 1) / (
                    tf + config.BM25_K1 * norm_factor
                )
                scores[doc_idx] += bm25_term

        return scores

    def save(self, artifacts_dir: Path) -> None:
        """Persist the index."""
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        with open(artifacts_dir / "inverted_index.json", "w") as f:
            json.dump(
                {
                    "postings": {
                        term: postings for term, postings in self.postings.items()
                    },
                    "doc_lengths": self.doc_lengths,
                    "N": self.N,
                },
                f,
            )

    @classmethod
    def load(cls, artifacts_dir: Path) -> "InvertedIndex":
        """Load persisted index."""
        with open(artifacts_dir / "inverted_index.json") as f:
            data = json.load(f)
        return cls(data["postings"], data["doc_lengths"], data["N"])


def tokenize(text: str) -> list[str]:
    """Tokenize text the same way as engine/text_descriptor.lexical_tokens.

    Lowercase + split on non-alphanumeric (except apostrophe)."""
    return re.findall(r"[a-z']+", text.lower())
