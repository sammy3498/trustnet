import os

from dotenv import load_dotenv
from tavily import TavilyClient

from backend.app.retrieval.search_provider import (
    SearchProvider,
    SearchResult,
)

load_dotenv()


class TavilySearchProvider(SearchProvider):
    def __init__(self):
        api_key = os.getenv("SEARCH_API_KEY")

        if not api_key:
            raise RuntimeError(
                "SEARCH_API_KEY is not configured."
            )

        self.client = TavilyClient(
            api_key=api_key
        )

    def search(
        self,
        query: str,
        max_results: int,
        end_date: str | None = None,
    ) -> list[SearchResult]:
        search_args = {
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }

        if end_date:
            search_args["end_date"] = end_date

        response = self.client.search(
            **search_args
        )

        results = []

        for item in response.get(
            "results",
            [],
        ):
            results.append(
                SearchResult(
                    title=item.get(
                        "title",
                        "",
                    ),
                    url=item.get(
                        "url",
                        "",
                    ),
                    snippet=item.get(
                        "content",
                        "",
                    ),
                )
            )

        return results