"""Single source of truth for tunable constants across the search engine."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "db" / "resonance.db"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

# -- Embeddings --------------------------------------------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_EMBEDDING_DIM = 384

# Continuous audio-feature columns that go into the similarity vector.
# duration_ms/key/mode/time_signature/popularity are excluded: they're
# categorical or a ranking signal, not a mood/timbre signal.
AUDIO_FEATURES = [
    "danceability", "energy", "valence", "tempo", "loudness",
    "speechiness", "acousticness", "instrumentalness", "liveness",
]

# Weights applied to the (already unit-normalized) text and audio sub-vectors
# before concatenation and a final unit-normalization pass, so cosine
# similarity between combined vectors reduces to a plain dot product.
TEXT_WEIGHT = 0.6
AUDIO_WEIGHT = 0.4

# -- Custom LSH ANN index -----------------------------------------------------
LSH_L = 8                    # number of independent hash tables
LSH_K = 12                   # hyperplanes per table (bucket key = k bits)
LSH_SEED = 42                # hyperplanes are generated once and persisted;
                              # this seed only matters the first time they're built
MULTIPROBE_MAX_RADIUS = 2    # Hamming-neighbor bucket widening when a query's
                              # own bucket doesn't return enough candidates
MIN_CANDIDATES = 50          # stop widening once we have at least this many
TOPK_OVERSAMPLE = 5          # candidate pool should be >= top_k * this before ranking

# As the catalog grows well past ~5k rows, raise LSH_K roughly with log2(N) to
# keep per-bucket occupancy low, and raise LSH_L to preserve recall.

# -- BM25 ---------------------------------------------------------------------
BM25_K1 = 1.5
BM25_B = 0.75

# -- Hybrid ranking ------------------------------------------------------------
W_COSINE = 0.65
W_BM25 = 0.25
W_POPULARITY = 0.10
RELAXED_PREDICATE_BONUS = 0.05

# -- Relational-filter relaxation ladder ---------------------------------------
MIN_POOL = 30                 # if SQL prefilter yields fewer ids than this, relax

# -- Recommend (track-to-track) ranking ------------------------------------------
# No BM25 term exists for recommend (there's no free-text query), so the
# 0.65/0.10 cosine:popularity ratio from text search is preserved after
# redistributing BM25's weight to sum to 1.0.
RECOMMEND_W_COSINE = 0.85
RECOMMEND_W_POPULARITY = 0.15
MAX_FAVORITES = 5
