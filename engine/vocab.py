"""Shared vocabulary for the catalog: genres, moods, and the mood<->audio-feature
heuristic. Imported by both the synthetic data generator (so generated mood tags
are consistent with these rules) and the NL query parser (so a query for "happy"
maps to the same audio-feature signal the data was generated to satisfy).
"""

GENRES = [
    "pop", "rock", "hip-hop", "electronic", "jazz", "classical",
    "country", "r&b", "reggae", "metal", "folk", "indie", "latin", "blues",
]

# Free-text words a query might use that map onto a canonical genre above.
GENRE_SYNONYMS = {
    "rap": "hip-hop",
    "hiphop": "hip-hop",
    "rnb": "r&b",
    "r&amp;b": "r&b",
    "edm": "electronic",
    "electro": "electronic",
    "dance": "electronic",
    "classic": "classical",
    "orchestral": "classical",
    "acoustic-folk": "folk",
    "hard rock": "rock",
}

MOODS = [
    "happy", "sad", "energetic", "chill", "aggressive", "romantic",
    "melancholic", "uplifting", "dreamy", "tense", "nostalgic", "playful",
]

# Each rule: a group of mood tags that share the same underlying audio-feature
# signal. `feature` is a column in `tracks`, `op` is ">" or "<", `threshold` is
# the value that side of the comparison should hold.
#
# generate_dataset.py uses these to bias which mood tags get attached to a
# track given its randomly rolled audio features. engine/query_parser.py uses
# them in reverse: if a query mentions one of these moods, it adds the same
# (feature, op, threshold) as a candidate SQL/ranking filter.
MOOD_AUDIO_HINTS = [
    (("happy", "uplifting", "playful"), "valence", ">", 0.6),
    (("sad", "melancholic", "nostalgic"), "valence", "<", 0.4),
    (("energetic", "aggressive", "tense"), "energy", ">", 0.7),
    (("chill", "dreamy"), "energy", "<", 0.4),
]
