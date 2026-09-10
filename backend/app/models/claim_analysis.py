from pydantic import (
    BaseModel,
    ConfigDict,
)


class ClaimAnalysisResponse(
    BaseModel
):
    normalized_claim: str
    is_clear: bool
    reason: str


class ClaimPreparationResponse(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    normalized_claim: str
    is_clear: bool
    reason: str
    search_query: str
    verification_queries: list[str]