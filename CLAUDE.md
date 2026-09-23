# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Resonance is a music search engine that uses hybrid ranking to find tracks matching natural language queries. It combines three ranking signals: text embeddings (from query + track metadata), Spotify-like audio features, and BM25 full-text relevance.

## Architecture

The project has three main layers:

**Data Pipeline** (`scripts/`):
- `generate_dataset.py`: Creates synthetic ~5k track dataset with genre-conditioned audio features and mood tags. Injects intentional defects to exercise validation.
- `load_and_validate.py`: Loads CSV into SQLite, validating all fields (ranges, nullability, duplicates). Keeps rejected rows in `rejected_rows` table for auditing instead of silently dropping them.
- `schema.sql`: Defines `tracks` (the main catalog) and `rejected_rows` (audit log). Includes CHECK constraints on all audio features to prevent invalid data in-DB.

**Embeddings & ANN** (`engine/`):
- `vocab.py`: Single source of truth for genres (14 types), moods (12 types), and mood-to-audio-feature heuristics. Used by both data generation (to bias mood tags) and query parsing (to translate "happy" into audio feature filters).
- `config.py`: All tunable hyperparameters for embeddings, LSH index, BM25, and hybrid ranking. Includes weights for the three signals (text 65%, BM25 25%, popularity 10%) and relaxation ladder for SQL prefilter.

**Data Storage** (`db/`):
- `resonance.db`: SQLite database. Index on genre and artist for efficient filtering. No embeddings persisted yet (planned).

## Common Development Tasks

**Set up / run the data pipeline:**
```bash
# Generate 5000 synthetic tracks with 2% corruption rate
python scripts/generate_dataset.py --rows 5000 --seed 42

# Load and validate into DB, showing rejection summary and genre distribution
python scripts/load_and_validate.py

# Load with custom paths
python scripts/load_and_validate.py --csv data/tracks.csv --db db/resonance.db
```

**Customize the dataset generation:**
- Adjust genre profiles (center points for audio features) in `generate_dataset.py:GENRE_PROFILES`
- Adjust mood-to-feature mappings in `engine/vocab.py:MOOD_AUDIO_HINTS` (these are used by the generator via `make_mood_tags`)
- Change defect rate: `--defect-rate 0.05` (default 0.02)
- Change duplicate rate: `--duplicate-rate 0.01` (default 0.005)

**Tune ranking and search:**
- Text vs. audio weighting: `engine/config.py:TEXT_WEIGHT`, `AUDIO_WEIGHT`
- Hybrid ranking weights: `W_COSINE` (vector similarity), `W_BM25` (full-text), `W_POPULARITY`
- LSH index params (for ANN recall-speed tradeoff): `LSH_L` (hash tables), `LSH_K` (hyperplanes per table)
- BM25 params: `BM25_K1`, `BM25_B`

**Inspect the database:**
```bash
sqlite3 db/resonance.db
# Show all tracks:
SELECT id, title, artist, genre, mood_tags, valence, energy FROM tracks LIMIT 10;

# Show rejected rows with errors:
SELECT row_num, errors FROM rejected_rows LIMIT 5;

# Genre distribution:
SELECT genre, COUNT(*) FROM tracks GROUP BY genre ORDER BY 2 DESC;
```

## Validation & Quality

All validation happens at load time (`load_and_validate.py`), enforced by:
- Python coercion and range checks before INSERT
- SQLite CHECK constraints in schema
- Duplicate ID detection (tracked in-memory during load)

The intent: bad data can't reach the DB, and rejections are audited (not silently dropped).

## Early-Stage Notes

- **Embeddings not yet computed/stored**: Config references sentence-transformers, but actual embedding pipeline is not yet built. Will need a `embeddings` table and bulk computation from `tracks`.
- **LSH index not yet built**: Config has LSH hyperparameters, but the index data structure is not yet persisted. Will be keyed by track ID.
- **Query parser not yet built**: `vocab.py:MOOD_AUDIO_HINTS` defines the mapping, but the query-to-SQL-filter translation doesn't exist yet.
- **API not yet built**: `api/` directory is empty. Will eventually expose query endpoint.

When adding these features, keep configuration in `engine/config.py` as the single source of truth, and update `vocab.py` if mood/genre taxonomy changes.
