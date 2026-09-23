"""Rule-based natural-language query parser.

Translates a free-text search query into structured signal the rest of the
engine can use: canonical genres/moods mentioned, audio-feature thresholds
they imply, and the untouched raw text (kept for the semantic embedding).
No LLM call - pure vocabulary/regex matching against `engine.vocab`.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from engine import vocab

AudioFilter = Tuple[str, str, float]

# Adjectives that aren't part of the shared genre/mood taxonomy in vocab.py -
# query-side-only heuristics, so they live here rather than in vocab.py.
# "chill" and "sad" are deliberately omitted: both are already canonical
# vocab.MOODS entries with their own vocab.MOOD_AUDIO_HINTS thresholds, so
# handling them here too would create a second source of truth for the same
# filter.
ADJECTIVE_RULES: Dict[str, AudioFilter] = {
    "upbeat":       ("valence", ">", 0.6),
    "fast":         ("tempo", ">", 120.0),
    "slow":         ("tempo", "<", 90.0),
    "acoustic":     ("acousticness", ">", 0.6),
    "instrumental": ("instrumentalness", ">", 0.5),
    "danceable":    ("danceability", ">", 0.6),
    "loud":         ("loudness", ">", -6.0),
    "quiet":        ("loudness", "<", -12.0),
}

# genre/synonym text -> canonical genre, longest keys first so a multi-word
# phrase is matched before any shorter substring it contains.
_GENRE_LOOKUP: Dict[str, str] = {g: g for g in vocab.GENRES}
_GENRE_LOOKUP.update(vocab.GENRE_SYNONYMS)
_GENRE_KEYS_BY_LENGTH = sorted(_GENRE_LOOKUP, key=len, reverse=True)


@dataclass
class ParsedQuery:
    raw_text: str
    genres: List[str] = field(default_factory=list)
    moods: List[str] = field(default_factory=list)
    audio_filters: List[AudioFilter] = field(default_factory=list)


def _match_genres(text_lower: str) -> List[str]:
    genres: List[str] = []
    for key in _GENRE_KEYS_BY_LENGTH:
        if re.search(r"\b" + re.escape(key) + r"\b", text_lower):
            canonical = _GENRE_LOOKUP[key]
            if canonical not in genres:
                genres.append(canonical)
    return genres


def _match_moods(tokens: set) -> List[str]:
    return [m for m in vocab.MOODS if m in tokens]


def _match_adjective_filters(tokens: set) -> List[AudioFilter]:
    return [ADJECTIVE_RULES[adj] for adj in ADJECTIVE_RULES if adj in tokens]


def _mood_filters(moods: List[str]) -> List[AudioFilter]:
    filters: List[AudioFilter] = []
    mood_set = set(moods)
    for mood_tuple, feature, op, threshold in vocab.MOOD_AUDIO_HINTS:
        if mood_set.intersection(mood_tuple):
            filters.append((feature, op, threshold))
    return filters


def _merge_filters(filters: List[AudioFilter]) -> List[AudioFilter]:
    merged: Dict[Tuple[str, str], float] = {}
    for feature, op, threshold in filters:
        key = (feature, op)
        if key not in merged:
            merged[key] = threshold
        elif op == ">":
            merged[key] = max(merged[key], threshold)
        else:  # "<"
            merged[key] = min(merged[key], threshold)
    return [(feature, op, threshold) for (feature, op), threshold in merged.items()]


def mood_audio_filters(moods: List[str]) -> List[AudioFilter]:
    """Translate canonical moods into merged (feature, op, threshold) filters.

    Used by callers that already have a mood and aren't going through parse_query()
    (e.g., engine.search.recommend)."""
    return _merge_filters(_mood_filters(moods))


def mood_target_features(mood: str | None) -> dict[str, float]:
    """Convert a mood to ReccoBeats target audio feature params.

    Maps a canonical mood to {'target_valence': X, 'target_energy': Y} for
    biasing ReccoBeats' /track/recommendation endpoint. Uses the threshold
    midpoint between the threshold and the open end of its range.

    Args:
        mood: One of vocab.MOODS, or None

    Returns:
        Dict with 'target_valence' and/or 'target_energy' keys, or empty dict
        if mood is None or unknown (maps to no bias, same as today).
    """
    if not mood:
        return {}

    targets = {}

    for mood_tuple, feature, op, threshold in vocab.MOOD_AUDIO_HINTS:
        if mood not in mood_tuple:
            continue

        if feature == "valence":
            if op == ">":
                targets["target_valence"] = (threshold + 1.0) / 2.0
            else:
                targets["target_valence"] = threshold / 2.0
        elif feature == "energy":
            if op == ">":
                targets["target_energy"] = (threshold + 1.0) / 2.0
            else:
                targets["target_energy"] = threshold / 2.0

    return targets


def derive_mood_tags(valence: float, energy: float) -> str:
    """Derive mood tags from real audio features using MOOD_AUDIO_HINTS.

    Reverse-checks audio features against MOOD_AUDIO_HINTS thresholds and
    returns any matching mood words as a comma-joined string.

    Args:
        valence: Audio feature (0-1)
        energy: Audio feature (0-1)

    Returns:
        Comma-joined mood tags, or "" if no matches.
    """
    matching_moods = []

    for mood_tuple, feature, op, threshold in vocab.MOOD_AUDIO_HINTS:
        value = valence if feature == "valence" else energy

        if (op == ">" and value > threshold) or (op == "<" and value < threshold):
            matching_moods.extend(mood_tuple)

    return ",".join(matching_moods) if matching_moods else ""


def parse_query(text: str) -> ParsedQuery:
    text_lower = text.lower()
    tokens = set(re.findall(r"[a-z']+", text_lower))

    genres = _match_genres(text_lower)
    moods = _match_moods(tokens)
    audio_filters = _merge_filters(
        _match_adjective_filters(tokens) + _mood_filters(moods)
    )

    return ParsedQuery(
        raw_text=text,
        genres=genres,
        moods=moods,
        audio_filters=audio_filters,
    )


if __name__ == "__main__":
    sample_queries = [
        "upbeat pop songs",
        "something sad and nostalgic",
        "rap and r&b",
        "hard rock anthems",
        "happy but also melancholic",
        "asdkfj qwerty",
    ]
    for query in sample_queries:
        print(f"{query!r} -> {parse_query(query)}")
