"""Bounded search recovery for insufficiently covered material propositions.

This module supplies queries, never verdicts. The recovered passages must pass
exactly the same LLM evidence-analysis and grounding gates as initial results.
"""
from __future__ import annotations
import re
from backend.app.retrieval.verification_plan import atomic_search_facets, _entities


def recovery_query(claim: str, result: dict) -> str | None:
    """Choose one missing proposition for a final, targeted search."""
    if result.get("label") != "Insufficient Evidence":
        return None
    evidence = result.get("evidence_analysis") or {}
    subclaims = evidence.get("subclaim_assessments") or []
    missing = [str(row.get("subclaim", "")).strip() for row in subclaims
               if row.get("status") == "insufficient" or row.get("coverage") in {"none", "partial"}]
    missing = [s for s in missing if len(s) >= 10]
    facets = atomic_search_facets(claim)
    target = missing[0] if missing else (facets[0] if len(facets) > 1 else claim)
    entities = _entities(claim)
    years = " ".join(dict.fromkeys(re.findall(r"\b(?:19|20)\d{2}\b", claim)))
    # If an attributed finding is not covered, search for the publisher's
    # actual conclusion or dated statement rather than the claim's assertion.
    if re.search(r"\b(review|study|research|statement|stated|announced|declared|quoted)\b", claim, re.I):
        subject = re.sub(r"\b(conclusively|proved|proves|always|never|every|none|all)\b", "",target, flags=re.I)
        return " ".join(f"{entities} {years} {subject[:130]} original publication author clarification full conclusion".split())[:250]
    return " ".join(f"{target[:170]} original authoritative primary source evidence".split())[:250]
