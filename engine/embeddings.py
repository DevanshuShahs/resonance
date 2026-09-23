"""Sentence-transformer embeddings and vector composition.

Loads the pretrained model lazily (importing this module doesn't download anything).
Combines text embeddings with normalized audio features into a hybrid vector."""

import numpy as np
from sentence_transformers import SentenceTransformer

from engine import config

_model = None


def get_model() -> SentenceTransformer:
    """Load and cache the pretrained model. Lazy singleton."""
    global _model
    if _model is None:
        _model = SentenceTransformer(config.MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts.

    Returns: ndarray of shape (N, config.TEXT_EMBEDDING_DIM), already
    unit-normalized per row (sentence-transformers does this by default)."""
    model = get_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def compose_vectors(
    text_vectors: np.ndarray, audio_vectors: np.ndarray
) -> np.ndarray:
    """Combine text and audio sub-vectors into a single hybrid vector.

    Args:
        text_vectors: (N, 384) unit-normalized
        audio_vectors: (N, 9) in [0,1] range (from audio_norm.apply)

    Returns: (N, 393) unit-normalized composite vectors.

    Process:
    1. Unit-normalize audio_vectors per row (all-zero rows stay zero).
    2. Apply TEXT_WEIGHT and AUDIO_WEIGHT to respective sub-vectors.
    3. Concatenate to (N, 393).
    4. L2-normalize the result so cosine similarity reduces to dot product."""

    # Unit-normalize audio, guarding against all-zero rows
    audio_norms = np.linalg.norm(audio_vectors, axis=1, keepdims=True)
    audio_norms[audio_norms == 0] = 1  # avoid division by zero
    audio_normalized = audio_vectors / audio_norms

    # Apply weights
    weighted_text = text_vectors * config.TEXT_WEIGHT
    weighted_audio = audio_normalized * config.AUDIO_WEIGHT

    # Concatenate
    combined = np.hstack([weighted_text, weighted_audio])

    # Final L2 normalization
    combined_norms = np.linalg.norm(combined, axis=1, keepdims=True)
    combined_norms[combined_norms == 0] = 1  # guard (shouldn't happen, but be safe)
    return combined / combined_norms


def query_audio_vector(audio_filters: list[tuple[str, str, float]]) -> np.ndarray:
    """Build a query audio vector from parsed audio filters.

    For each (feature, op, threshold) filter: sets the feature's slot to +1.0
    (if op is ">") or -1.0 (if op is "<"), encoding "want high" vs "want low".
    Unmentioned features stay 0. This signed representation makes "<" constraints
    work under dot-product similarity: a negative slot penalizes high values
    and rewards low ones, which is the inverse of the dot product's natural
    preference for magnitude.

    Args:
        audio_filters: list of (feature, op, threshold) tuples from ParsedQuery

    Returns: shape (9,) array, one slot per config.AUDIO_FEATURES, values in
        {-1, 0, +1}. If audio_filters is empty, returns all zeros."""
    vector = np.zeros(len(config.AUDIO_FEATURES), dtype=np.float32)
    for feature, op, _ in audio_filters:
        try:
            idx = config.AUDIO_FEATURES.index(feature)
            vector[idx] = 1.0 if op == ">" else -1.0
        except (ValueError, IndexError):
            pass
    return vector
