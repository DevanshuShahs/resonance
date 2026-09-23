#!/usr/bin/env python3
"""Test the full search engine with sample queries."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import search, store


def main():
    print("Loading search engine...")
    engine_store = store.load()
    print(f"✓ Loaded {len(engine_store.track_ids)} tracks")
    print(f"✓ LSH index: {engine_store.lsh_index.L} tables")
    print(f"✓ Inverted index: {len(engine_store.inverted_index.postings)} terms\n")

    queries = [
        "upbeat pop songs",
        "sad and melancholic jazz",
        "fast energetic electronic",
        "happy indie vibes",
        "chill acoustic folk",
        "aggressive heavy metal",
    ]

    for query in queries:
        print(f"Query: {query!r}")
        results = search.search(engine_store, query, top_k=5)

        if not results:
            print("  No results.\n")
            continue

        for result in results:
            print(
                f"  {result.rank}. {result.title:30s} by {result.artist:20s} "
                f"[{result.genre:12s}] {result.mood_tags:30s}"
            )
            print(
                f"      score={result.score:.4f} "
                f"(cosine={result.cosine:.3f} + bm25={result.bm25:.3f} + "
                f"popularity={result.popularity_norm:.3f} + bonus={result.relaxation_bonus:.3f}) "
                f"[{result.filter_level}]"
            )
        print()


if __name__ == "__main__":
    main()
