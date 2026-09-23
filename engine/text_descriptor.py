"""Extract semantic and lexical signals from a track for embedding and indexing."""

import re


def semantic_text(track: dict) -> str:
    """Text for semantic embedding: genre + mood_tags only.

    Title/artist are synthetic random names with no real semantic content and
    would just add embedding noise, so they're excluded. Genre and mood_tags
    carry the human-interpretable meaning of a track."""
    moods = track.get("mood_tags", "").strip()
    genre = track.get("genre", "").strip()

    if moods:
        mood_list = [m.strip() for m in moods.split("|") if m.strip()]
        if mood_list:
            return f"{genre} music that feels {', '.join(mood_list)}"
    return f"{genre} music"


def lexical_tokens(track: dict) -> list[str]:
    """Tokens for lexical indexing (BM25): title + artist + genre + mood_tags.

    Lowercased and split into words. Used by a future inverted-index builder,
    not by this module."""
    tokens = []
    for field in ["title", "artist", "genre", "mood_tags"]:
        value = track.get(field, "").strip()
        if value:
            if field == "mood_tags":
                value = value.replace("|", " ")
            words = re.findall(r"[a-z']+", value.lower())
            tokens.extend(words)
    return tokens
