from typing import Protocol


class SearchResult:
    def __init__(
        self,
        title: str,
        url: str,
        snippet: str,
    ):
        self.title = title
        self.url = url
        self.snippet = snippet


class SearchProvider(Protocol):
    def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[SearchResult]:
        ...