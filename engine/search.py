"""Full search pipeline: SQL prefilter + LSH + BM25 + ranking.

Loads a EngineStore once (at API startup), then executes search queries by:
1. Parsing the natural-language query
2. Embedding the query + composing with audio signal
3. Running SQL prefilter with relaxation ladder
4. Retrieving LSH candidates, bounded by SQL allowlist
5. Computing cosine similarity over candidates
6. Computing BM25 scores over candidates
7. Blending scores: W_COSINE*cosine + W_BM25*bm25 + W_POPULARITY*popularity + relaxation_bonus
8. Returning top-K ranked results
"""

import heapq
from dataclasses import dataclass

import numpy as np

from engine import config, embeddings, inverted_index, query_parser, sql_filters, store


@dataclass
class ScoredResult:
    rank: int
    track_id: str
    title: str
    artist: str
    genre: str
    mood_tags: str
    score: float
    cosine: float
    bm25: float
    popularity_norm: float
    relaxation_bonus: float
    filter_level: str


@dataclass
class RecommendedResult:
    rank: int
    track_id: str
    title: str
    artist: str
    genre: str
    mood_tags: str
    score: float
    cosine: float
    popularity_norm: float
    relaxation_bonus: float
    filter_level: str


def search(
    engine_store: "store.EngineStore", query_text: str, top_k: int = 10
) -> list[ScoredResult]:
    """Execute a full search query.

    Args:
        engine_store: loaded EngineStore from engine.store.load()
        query_text: free-text search query
        top_k: number of results to return

    Returns: list of ScoredResult, ranked by final score descending
    """

    # 1. Parse query
    parsed = query_parser.parse_query(query_text)

    # 2. SQL prefilter with relaxation
    prefilter = sql_filters.prefilter_ids(engine_store.conn, parsed)
    sql_allowed_indices = {
        i for i, tid in enumerate(engine_store.track_ids) if tid in prefilter.ids
    }

    if not sql_allowed_indices:
        return []

    # 3. Embed query + compose vector
    query_text_vec = embeddings.embed_texts([parsed.raw_text])[0]
    query_audio_vec = embeddings.query_audio_vector(parsed.audio_filters)
    query_vector = embeddings.compose_vectors(
        np.array([query_text_vec]), np.array([query_audio_vec])
    )[0]

    # 4. LSH retrieval
    lsh_candidates = engine_store.lsh_index.query(query_vector)
    scoped_lsh = lsh_candidates & sql_allowed_indices

    # If LSH+SQL intersection is small, fall back to brute-force within SQL allowlist
    needed = max(config.MIN_CANDIDATES, top_k * config.TOPK_OVERSAMPLE)
    if len(scoped_lsh) < needed:
        scoped_lsh = sql_allowed_indices

    candidate_indices = sorted(scoped_lsh)

    # 5. Cosine similarity over candidates
    cosine_scores = np.dot(engine_store.embeddings[candidate_indices], query_vector)

    # 6. BM25 scores
    query_tokens = inverted_index.tokenize(parsed.raw_text)
    bm25_raw = engine_store.inverted_index.bm25_scores(
        query_tokens, set(candidate_indices)
    )

    # Normalize BM25 to [0, 1]
    bm25_values = [bm25_raw.get(idx, 0.0) for idx in candidate_indices]
    bm25_max = max(bm25_values) if bm25_values else 1.0
    if bm25_max == 0:
        bm25_max = 1.0
    bm25_norm = {
        idx: bm25_raw.get(idx, 0.0) / bm25_max for idx in candidate_indices
    }

    # 7. Final blended score
    results = []
    for rank, (score_idx, cosine_score) in enumerate(
        sorted(
            enumerate(cosine_scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k],
        start=1,
    ):
        actual_idx = candidate_indices[score_idx]
        track_id = engine_store.track_ids[actual_idx]
        track = engine_store.tracks[track_id]

        bm25_score = bm25_norm[actual_idx]
        popularity_score = float(track["popularity"]) / 100.0
        relaxation_bonus = (
            config.RELAXED_PREDICATE_BONUS
            * sql_filters.satisfied_dropped_count(track, prefilter)
        )

        final_score = (
            config.W_COSINE * cosine_score
            + config.W_BM25 * bm25_score
            + config.W_POPULARITY * popularity_score
            + relaxation_bonus
        )

        results.append(
            ScoredResult(
                rank=rank,
                track_id=track_id,
                title=track["title"],
                artist=track["artist"],
                genre=track["genre"],
                mood_tags=track["mood_tags"],
                score=final_score,
                cosine=float(cosine_score),
                bm25=bm25_score,
                popularity_norm=popularity_score,
                relaxation_bonus=float(relaxation_bonus),
                filter_level=prefilter.level,
            )
        )

    # Re-rank by final score
    results.sort(key=lambda r: r.score, reverse=True)
    for rank, result in enumerate(results, start=1):
        result.rank = rank

    return results


def recommend(
    engine_store: "store.EngineStore",
    favorite_track_ids: list[str],
    mood: str | None = None,
    top_k: int = 10,
) -> list[RecommendedResult]:
    """Recommend tracks similar to a set of favorite tracks, optionally biased by mood.

    Args:
        engine_store: loaded EngineStore from engine.store.load()
        favorite_track_ids: 1-5 track IDs to build recommendations around
        mood: optional mood to bias the recommendation (one of vocab.MOODS)
        top_k: number of results to return

    Returns: list of RecommendedResult, ranked by final score descending
    """

    # 1. Validate and dedupe favorites
    ids = list(dict.fromkeys(favorite_track_ids))
    if not ids or len(ids) > config.MAX_FAVORITES:
        raise ValueError(
            f"Must provide between 1 and {config.MAX_FAVORITES} favorite track ids; "
            f"got {len(ids)}"
        )

    unknown = [tid for tid in ids if tid not in engine_store.track_id_to_idx]
    if unknown:
        raise ValueError(f"Unknown track id(s): {unknown}")

    # 2. SQL prefilter with mood-derived filters and relaxation ladder
    parsed = query_parser.ParsedQuery(
        raw_text="",
        genres=[],
        moods=[mood] if mood else [],
        audio_filters=query_parser.mood_audio_filters([mood] if mood else []),
    )
    prefilter = sql_filters.prefilter_ids(engine_store.conn, parsed)
    sql_allowed_indices = {
        i for i, tid in enumerate(engine_store.track_ids) if tid in prefilter.ids
    }

    if not sql_allowed_indices:
        return []

    # 3. Build candidate pool excluding favorites
    favorite_indices = [engine_store.track_id_to_idx[tid] for tid in ids]
    favorite_index_set = set(favorite_indices)
    sql_allowed_indices = sql_allowed_indices - favorite_index_set

    if not sql_allowed_indices:
        return []

    # 4. Taste vector: mean of favorite embeddings, L2-normalized
    centroid = engine_store.embeddings[favorite_indices].mean(axis=0)
    norm = np.linalg.norm(centroid)
    taste_vector = centroid / (norm if norm != 0 else 1)

    # 5. LSH retrieval with SQL scoping and brute-force fallback
    lsh_candidates = engine_store.lsh_index.query(taste_vector)
    scoped_lsh = lsh_candidates & sql_allowed_indices

    needed = max(config.MIN_CANDIDATES, top_k * config.TOPK_OVERSAMPLE)
    if len(scoped_lsh) < needed:
        scoped_lsh = sql_allowed_indices

    candidate_indices = sorted(scoped_lsh)

    # 6. Cosine similarity over candidates
    cosine_scores = np.dot(engine_store.embeddings[candidate_indices], taste_vector)

    # 7. Final blended score (no BM25 for recommendations)
    results = []
    for rank, (score_idx, cosine_score) in enumerate(
        sorted(
            enumerate(cosine_scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k],
        start=1,
    ):
        actual_idx = candidate_indices[score_idx]
        track_id = engine_store.track_ids[actual_idx]
        track = engine_store.tracks[track_id]

        popularity_score = float(track["popularity"]) / 100.0
        relaxation_bonus = (
            config.RELAXED_PREDICATE_BONUS
            * sql_filters.satisfied_dropped_count(track, prefilter)
        )

        final_score = (
            config.RECOMMEND_W_COSINE * cosine_score
            + config.RECOMMEND_W_POPULARITY * popularity_score
            + relaxation_bonus
        )

        results.append(
            RecommendedResult(
                rank=rank,
                track_id=track_id,
                title=track["title"],
                artist=track["artist"],
                genre=track["genre"],
                mood_tags=track["mood_tags"],
                score=final_score,
                cosine=float(cosine_score),
                popularity_norm=popularity_score,
                relaxation_bonus=float(relaxation_bonus),
                filter_level=prefilter.level,
            )
        )

    # Re-rank by final score
    results.sort(key=lambda r: r.score, reverse=True)
    for rank, result in enumerate(results, start=1):
        result.rank = rank

    return results
