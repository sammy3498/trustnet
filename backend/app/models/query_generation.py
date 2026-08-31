from pydantic import BaseModel


class QueryGenerationResponse(BaseModel):
    query: str