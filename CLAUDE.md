# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Resonance is a music discovery platform with **two parallel, independently-functioning search/ranking systems** that both live behind the same FastAPI app:

1. **Live ReccoBeats-backed flow** (`GET /tracks/search`, `POST /recommend`) - queries the live ReccoBeats API (`https://api.reccobeats.com/v1`, no auth) for typeahead search and recommendations, so catalog coverage isn't limited to a pre-downloaded dataset.
2. **Local hybrid-search engine** (`POST /search`) - a from-scratch IR pipeline (text+audio embeddings, hand-built LSH ANN, hand-built BM25) running against a static ~5k-track synthetic dataset persisted in SQLite + `artifacts/`.

These were built and are documented separately; don't assume changes to one affect the other. `plans/live-reccobeats-search.md` describes the design for #1, but its own header claim ("planned, not yet implemented") is now stale - it **is** implemented (see `api/routes.py`, `engine/reccobeats_client.py`, `engine/recommendation_engine.py`, `engine/hybrid_ranking.py`). Treat that plan file as historical design rationale, not a to-do list.

## Setup & Common Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start the API (loads the local engine at startup even though /tracks/search
# and /recommend don't use it - see api/main.py lifespan hook)
python -m uvicorn api.main:app --reload --port 8000

# Frontend (separate Next.js app, own CLAUDE.md/AGENTS.md - see frontend/)
cd frontend && npm run dev   # http://localhost:3000, expects API on :8000
```

Local-engine smoke test / benchmark:
```bash
python scripts/test_search.py   # 6 canned queries, full score breakdown
python scripts/bench_ann.py     # LSH vs brute-force: latency, candidates, Jaccard recall
```

Rebuilding the local dataset/engine from scratch:
```bash
python scripts/generate_dataset.py --rows 5000 --seed 42   # synthetic catalog -> CSV
python scripts/load_and_validate.py                          # CSV -> SQLite, validated
python scripts/build_embeddings.py                           # embeds + builds LSH/BM25 artifacts
```
`scripts/fetch_reccobeats_tracks.py` / `fetch_spotify_tracks.py` are the (now superseded for live search) original bulk-download scripts used to seed the local dataset from real catalogs.

## Architecture

### `engine/` - local hybrid-search IR pipeline (serves `POST /search` only)

- `vocab.py`: single source of truth for the 14 genres, 12 moods, and `MOOD_AUDIO_HINTS` (mood -> audio-feature threshold heuristics, e.g. "happy" -> valence > 0.6). Used by both the synthetic data generator and the query parser/mood-tag derivation - keep this in sync if the taxonomy changes.
- `config.py`: tunable constants for embeddings, LSH, BM25, and the two ranking pipelines. **Note:** `RECOMMEND_W_COSINE`/`RECOMMEND_W_POPULARITY` here are dead weight - the actual `/recommend` scoring weights (0.35/0.20/0.15/0.10/0.20) are hardcoded in `hybrid_ranking.hybrid_rerank()`, not read from this file. If tuning recommend ranking, edit `hybrid_ranking.py` directly.
- `text_descriptor.py` + `embeddings.py`: build the 384-dim MiniLM text embedding (genre + mood tags only - titles/artists are synthetic-random and excluded as noise) and the audio-feature vector, weight them (`TEXT_WEIGHT`/`AUDIO_WEIGHT`), concatenate, and unit-normalize so cosine similarity reduces to a dot product.
- `audio_norm.py`: computes/persists min-max stats for the 9 continuous audio features.
- `lsh_index.py`: random-hyperplane LSH with Hamming-neighbor multiprobe (widens search radius until `MIN_CANDIDATES` is met) - the from-scratch ANN index, no FAISS/hnswlib/annoy.
- `inverted_index.py`: hand-built BM25 (k1=1.5, b=0.75) over title/artist/genre/mood text.
- `query_parser.py`: rule-based NL parser (no LLM) - extracts genre/mood/adjective keywords from a query and derives mood tags from real valence/energy values (used by the live flow too, via `derive_mood_tags`).
- `sql_filters.py`: turns parsed query filters into a SQL WHERE clause with an automatic **relaxation ladder**: full filters -> genre-only -> unrestricted, triggered when the prefilter yields fewer than `MIN_POOL` rows. Dropped predicates become a `+0.05` score bonus if a candidate happens to satisfy them anyway.
- `store.py`: `EngineStore` loads all `artifacts/` + the DB once at startup and runs an alignment guard (fails loudly if the DB and embeddings have drifted out of sync - rerun `build_embeddings.py` after any DB change).
- `search.py`: orchestrates the full local pipeline (SQL prefilter -> LSH candidate retrieval -> BM25 scoring -> blended ranking).
- Final local-search score: `0.65 * cosine + 0.25 * bm25_norm + 0.10 * popularity_norm + relaxation_bonus` (weights in `config.py`).

### Live ReccoBeats flow (serves `GET /tracks/search`, `POST /recommend`)

- `reccobeats_client.py`: stateless HTTP client (module-level `requests.Session`, 8s timeout) wrapping three ReccoBeats endpoints - `track/search`, `track/recommendation`, `audio-features` (batched), `track` (batched metadata). All failures raise `ReccoBeatsError`, caught in `api/routes.py` and turned into a `502`.
- `recommendation_engine.py`: for `/recommend`, does **not** use ReccoBeats' own `/track/recommendation` endpoint for final ranking - instead it generates search queries from favorite tracks' title/artist/mood (`generate_search_queries`), pulls a candidate pool via `reccobeats_client.search_tracks`, fetches audio features for candidates, and hands off to `hybrid_ranking.hybrid_rerank` for scoring.
- `hybrid_ranking.py`: reranks candidates on 5 signals - audio similarity to favorites' centroid (35%), popularity alignment (20%), artist-freshness/discovery (20%), acoustic-profile alignment (15%), duration alignment (10%). These weights are local constants in this file, not `config.py` (see note above).
- Mood tags for live results are *derived* (not stored) from each track's real valence/energy via `query_parser.derive_mood_tags`, reusing `vocab.MOOD_AUDIO_HINTS` in reverse - ReccoBeats has no genre/mood field, so genre is dropped entirely from this flow's schemas.

### `api/`

- `main.py`: FastAPI app with a lifespan hook that loads `EngineStore` once at startup (needed only for `/search`) and CORS restricted to the Next.js dev origin (`localhost:3000`).
- `routes.py`: `GET /tracks/search`, `GET /moods`, `POST /search` (local engine), `POST /recommend` (live ReccoBeats + local hybrid rerank).
- `schemas.py`: Pydantic models - note `TrackSummarySchema`/`RecommendedResultSchema` (live flow) intentionally carry no genre/score/cosine/filter_level fields, since nothing computes those for live data; `SearchResultSchema` (local flow) carries full score breakdown.

### Data & Artifacts (local engine only)

- `db/resonance.db`: SQLite - `tracks` (main catalog, CHECK constraints on all audio-feature ranges) and `rejected_rows` (audit log of validation failures, never silently dropped).
- `artifacts/`: persisted `embeddings.npy`, `hyperplanes.npy`, `buckets.json` (LSH), `inverted_index.json` (BM25), `norm_stats.json`, `track_ids.json` - all produced by `scripts/build_embeddings.py` and loaded by `EngineStore`.

### `frontend/`

Separate Next.js app (own git repo, own `CLAUDE.md`/`AGENTS.md`/`README.md` - don't duplicate its docs here). Talks to the API on `localhost:8000`; UI flow is search-for-favorites -> pick mood -> get recommendations, rendering mood tags, popularity, and a Spotify link per result.

## Validation & Quality (local dataset pipeline)

All validation happens at load time (`load_and_validate.py`): Python range/type coercion, SQLite CHECK constraints, and in-memory duplicate-ID detection during load - bad rows are rejected and audited in `rejected_rows`, never silently dropped.
