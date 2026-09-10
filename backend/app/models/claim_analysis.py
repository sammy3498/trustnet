from pydantic import BaseModel


class ClaimAnalysisResponse(BaseModel):
    normalized_claim: str
    is_clear: bool
    reason: str

class ClaimPreparationResponse(BaseModel):
    normalized_claim: str
    is_clear: bool
    reason: str
    search_query: str
    verification_queries: list[str] = []