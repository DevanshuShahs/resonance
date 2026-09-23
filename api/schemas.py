"""Pydantic request/response schemas for the search API."""

from pydantic import BaseModel, Field, field_validator

from engine import vocab


class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural-language search query")
    top_k: int = Field(default=10, description="Number of results to return")


class SearchResultSchema(BaseModel):
    rank: int
    track_id: str
    title: str
    artist: str
    genre: str
    mood_tags: str
    score: float = Field(..., description="Final blended score")
    cosine: float = Field(..., description="Semantic similarity (0-1)")
    bm25: float = Field(..., description="BM25 full-text score (0-1)")
    popularity_norm: float = Field(..., description="Normalized popularity (0-1)")
    relaxation_bonus: float = Field(..., description="Bonus for satisfied dropped filters")
    filter_level: str = Field(..., description="SQL filter level used: 'full', 'genre_only', or 'unrestricted'")


class SearchResponse(BaseModel):
    query: str
    filter_level: str
    results: list[SearchResultSchema]


class TrackSummarySchema(BaseModel):
    track_id: str
    title: str
    artist: str
    mood_tags: str
    spotify_url: str | None = None


class TrackSearchResponse(BaseModel):
    query: str
    results: list[TrackSummarySchema]


class MoodsResponse(BaseModel):
    moods: list[str]


class RecommendRequest(BaseModel):
    favorite_track_ids: list[str] = Field(
        ..., min_length=1, max_length=5,
        description="1-5 track IDs selected from the catalog",
    )
    mood: str | None = Field(default=None, description="One of the 12 canonical moods, or null")
    top_k: int = Field(default=10, ge=1, le=50)

    @field_validator("mood")
    @classmethod
    def _mood_must_be_known(cls, v):
        if v is not None and v not in vocab.MOODS:
            raise ValueError(f"unknown mood: {v!r}")
        return v


class RecommendedResultSchema(BaseModel):
    rank: int
    track_id: str
    title: str
    artist: str
    mood_tags: str
    popularity: int = Field(..., description="Track popularity (0-100)")
    spotify_url: str | None = None


class RecommendResponse(BaseModel):
    favorite_track_ids: list[str]
    mood: str | None
    results: list[RecommendedResultSchema]
