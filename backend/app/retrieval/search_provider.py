from typing import Protocol
from dataclasses import dataclass
from typing import Protocol


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str

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
        max_results: int,
        end_date: str | None = None,
    ) -> list[SearchResult]:
        ...


class SearchProvider(Protocol):
    def search(
        self,
        query: str,
        max_results: int,
        end_date: str | None = None,
    ) -> list[SearchResult]:
        ...