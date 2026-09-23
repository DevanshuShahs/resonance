# Live ReccoBeats-backed search & recommendations

> Status: planned, not yet implemented. Saved here so future work sessions can pick this up
> without redoing the ReccoBeats API investigation below.

## Context

Resonance's recommendation frontend (search favorites → pick mood → get recommendations) currently runs entirely against a local snapshot: 1322 tracks fetched once from ReccoBeats into `db/resonance.db`, then embedded offline into `artifacts/` (sentence-transformer text vectors + audio vectors, LSH index, BM25 index). This caps song coverage at whatever was pre-downloaded and requires re-running the fetch+embed pipeline to grow the catalog.

The user asked for a way to search live instead of pre-downloading "millions" of songs, and confirmed the priority is **coverage** (search ReccoBeats' full catalog, not just 1322 tracks) and **storage** (no local DB/embeddings needed for this flow).

Investigation of the live ReccoBeats API (no auth required) found it's more capable than the original fetch script assumed:
- `GET /v1/track/search?searchText=&limit=` — live typeahead search
- `GET /v1/track/recommendation?seeds=<id1>,<id2>,...&size=N&target_valence=X&target_energy=Y` — **ReccoBeats' own similarity-based recommendation engine**, and it responds to Spotify-style `target_*` audio-feature params (confirmed live: `target_valence=0.1&target_energy=0.1` measurably changes ranking vs. no targets).
- `GET /v1/audio-features?ids=id1,id2,...` — **batched** audio-feature lookup for up to at least 25 ids in one call (confirmed live).
- No endpoint returns genre or mood tags for a track under any circumstance (search, track detail, artist detail all checked).

This means the live redesign does **not** need to reimplement the local hybrid-embedding + LSH + cosine pipeline per request — ReccoBeats' `/track/recommendation` endpoint replaces that entirely, including mood-biasing via `target_valence`/`target_energy`. The only local computation left is deriving mood tags for display, which can reuse the existing `vocab.MOOD_AUDIO_HINTS` thresholds against real audio features fetched from the batch endpoint.

**Decisions confirmed with the user:**
- The old local-corpus engine (`db/resonance.db`, `artifacts/`, `POST /search`, and the `engine/` modules that only serve it — `store.py`, `sql_filters.py`, `lsh_index.py`, `inverted_index.py`, `embeddings.py`, `audio_norm.py`, `text_descriptor.py`, `scripts/build_embeddings.py`) is left **fully in place and untouched** — not wired into the new live flow, not deleted. `api/main.py` keeps loading `EngineStore` at startup so `POST /search` keeps working.
- Mood tags must be kept in the UI. Since ReccoBeats provides no mood data, mood tags are **derived** from each track's real `valence`/`energy` (fetched via the batch audio-features endpoint) using the same threshold logic already in `vocab.MOOD_AUDIO_HINTS`, run in reverse.
- Genre has no live equivalent and is dropped from the API/UI for the live flow (search typeahead + recommend results).

## Recommended Approach

### 1. New module: `engine/reccobeats_client.py`

Thin HTTP client, no local state, module-level `requests.Session()` for connection reuse. Three functions:

```python
def search_tracks(query: str, limit: int) -> list[dict]:
    # GET {BASE}/track/search?searchText=&limit= -> response["content"]

def get_recommendations(seed_ids: list[str], size: int, target_features: dict[str, float]) -> list[dict]:
    # GET {BASE}/track/recommendation?seeds=<comma-joined>&size=&target_valence=&target_energy=...
    # -> response["content"]

def get_audio_features_batch(track_ids: list[str]) -> dict[str, dict]:
    # GET {BASE}/audio-features?ids=<comma-joined> -> {id: features_dict, ...}
```

All three raise `ReccoBeatsError` (new exception class in the same module) on non-2xx response, timeout (8s), or network error — never let a raw `requests` exception or 500 leak to the API layer. Reuses the exact endpoint shapes already proven working in `scripts/fetch_reccobeats_tracks.py` (camelCase fields: `trackTitle`, `durationMs`, `artists[].name`, `href`).

### 2. Mood ↔ audio-feature helpers (extend `engine/query_parser.py`)

Two new functions alongside the existing `mood_audio_filters()`, both driven by the existing `vocab.MOOD_AUDIO_HINTS` table (single source of truth preserved, per project convention):

```python
def mood_target_features(mood: str | None) -> dict[str, float]:
    """Mood -> {'target_valence': .., 'target_energy': ..} for ReccoBeats' recommendation params.
    Midpoint of the threshold and the open end of its range, e.g. valence > 0.6 -> target 0.8.
    Unknown/None mood (e.g. 'romantic', which has no hint entry) -> {} (no bias, same as today)."""

def derive_mood_tags(valence: float, energy: float) -> str:
    """Reverse-check real audio features against MOOD_AUDIO_HINTS thresholds;
    return comma-joined matching mood words, or "" if none match."""
```

### 3. API layer — rewire two endpoints, add no new ones

**`api/routes.py`**

- `GET /tracks/search`: replace the SQLite `LIKE` query with `reccobeats_client.search_tracks(q, limit)`, batch-fetch audio features for the returned ids via `get_audio_features_batch`, derive `mood_tags` per track via `derive_mood_tags`, build `TrackSummarySchema` (drop `genre`, add `spotify_url` from each track's `href`).
- `POST /recommend`: drop the `engine.search.recommend()` call and the "unknown id" pre-check against `engine_store.tracks` (ids now come live from ReccoBeats, not a local table). New flow: `target_features = query_parser.mood_target_features(req.mood)` → `reccobeats_client.get_recommendations(req.favorite_track_ids, req.top_k, target_features)` → batch audio-features on the result ids → derive mood_tags → build `RecommendedResultSchema` in ReccoBeats' returned order (`rank` = 1-based position). Wrap `reccobeats_client.ReccoBeatsError` → `HTTPException(502, "Music service temporarily unavailable")`; if ReccoBeats rejects the seeds (bad ids), surface as `HTTPException(422, ...)`.
- `GET /moods`, `POST /search` — **unchanged**.

**`api/schemas.py`**
- `TrackSummarySchema`: `track_id, title, artist, mood_tags, spotify_url: str | None` (remove `genre`).
- `RecommendedResultSchema`: `rank, track_id, title, artist, mood_tags, popularity: int, spotify_url: str | None` (remove `genre`, `score`, `cosine`, `popularity_norm`, `relaxation_bonus`, `filter_level` — nothing computes those anymore, so no fabricated numbers get returned).
- `RecommendResponse`: `favorite_track_ids, mood, results` (drop `filter_level`).
- `RecommendRequest`, `MoodsResponse`, `TrackSearchResponse`, `SearchRequest`/`SearchResultSchema`/`SearchResponse` (used by untouched `/search`) — unchanged.

**`api/main.py`** — unchanged (still loads `EngineStore` at startup for `/search`; `reccobeats_client` is stateless, no startup wiring needed).

### 4. Frontend — follow the schema changes through

- `app/lib/types.ts`: `TrackSummary` drops `genre`, adds `spotify_url?: string | null`; `RecommendedResult` drops `genre`/`cosine`/`popularity_norm`/`relaxation_bonus`/`filter_level`/`score`, keeps `rank/track_id/title/artist/mood_tags`, adds `popularity: number`, `spotify_url?: string | null`.
- `app/components/SongPicker.tsx`: dropdown line `{track.artist} • {track.genre}` → `{track.artist}`.
- `app/components/ResultsList.tsx`: remove the genre pill and the score-bar/semantic%/popularity% block (no longer computed); keep the `mood_tags` pill (now real data); add a `popularity` badge and a "▶ Listen on Spotify" link (`track.spotify_url`, `target="_blank"`) — replaces the removed score visualization with something a user can actually act on.
- `app/lib/api.ts`, `app/page.tsx` — no functional changes, only type flow-through.

## Critical Files
- `engine/reccobeats_client.py` — new
- `engine/query_parser.py` — add `mood_target_features()`, `derive_mood_tags()`
- `api/routes.py` — rewire `GET /tracks/search`, `POST /recommend`
- `api/schemas.py` — trim `TrackSummarySchema`, `RecommendedResultSchema`, `RecommendResponse`
- `frontend/.claude/worktrees/resonance-frontend/app/lib/types.ts`, `.../components/SongPicker.tsx`, `.../components/ResultsList.tsx`

## Verification

**Backend** (`source .venv/bin/activate && uvicorn api.main:app --reload --port 8000`):
```bash
# live typeahead, no local DB involved — try a track NOT in the local 1322
curl -s "http://localhost:8000/tracks/search?q=<obscure or very recent song title>&limit=5" | python3 -m json.tool
# expect: results with mood_tags populated, spotify_url present, no genre field

curl -s -X POST http://localhost:8000/recommend -H 'Content-Type: application/json' \
  -d '{"favorite_track_ids": ["<id from search above>"], "mood": "happy", "top_k": 10}' | python3 -m json.tool
# rerun with "mood": "sad" on the same favorite — confirm mood_tags/ranking visibly shift

# error path: garbage id -> clean 422/502, not a stack trace
curl -s -X POST http://localhost:8000/recommend -d '{"favorite_track_ids": ["not-a-real-id"], "mood": "happy"}'

# confirm /search still works untouched (local engine unaffected)
curl -s -X POST http://localhost:8000/search -d '{"query": "upbeat pop", "top_k": 5}' | python3 -m json.tool
```

**Frontend** (`cd frontend/.claude/worktrees/resonance-frontend && npm run dev` → `http://localhost:3000`): search for a song outside the local 1322-track set and confirm it's found; pick 1-5 favorites, pick a mood, submit, confirm result cards show real mood tags, a working "Listen on Spotify" link, and no genre pill; change mood and resubmit, confirm results and mood tags shift; stop the network/kill the process briefly and resubmit to confirm a clean error state instead of a hang or crash.
