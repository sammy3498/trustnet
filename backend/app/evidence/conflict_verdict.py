"""Ground a disputed/misleading verdict in evidence, not mere topical disagreement.

This module checks citations and structural sufficiency. It does not independently
prove factual correctness; the LLM still needs to assess the cited material.
"""
from backend.app.evidence.domain import get_publisher_domain


def grounded_mixed_evidence(analysis: dict, evidence: list) -> dict:
    """Return evidence-grounding metadata for a mixed or cherry-picked verdict.

    A material conflict needs (a) opposing direct cited stances about the claim,
    (b) an explicitly mixed directly relevant passage, or (c) a directly sourced
    material qualifier that makes a supported statement misleading. The evidence
    must include at least one high/medium-authority publisher; one well-sourced
    fact check can establish a material context omission.
    """
    by_url = {doc.url.rstrip("/"): doc for doc in evidence}
    rows = []
    publishers = set()
    reliable = set()
    for item in analysis.get("evidence_assessments", []):
        doc = by_url.get(str(item.get("url", "")).rstrip("/"))
        if (doc is None or item.get("directness") not in {"direct", "partial"}
                or item.get("relevance") not in {"high", "medium"}
                or item.get("stance") not in {"supports", "contradicts", "mixed"}):
            continue
        rows.append((item, doc))
        key = get_publisher_domain(doc.source or doc.url) or doc.url
        publishers.add(key)
        if doc.quality in {"high", "medium"}:
            reliable.add(key)
    stances = {item.get("stance") for item, _ in rows}
    cited_conflict = ("mixed" in stances or {"supports", "contradicts"} <= stances)
    context_subclaim = any(x.get("material_context_conflict") for x in analysis.get("subclaim_assessments", []))
    cited_qualifier = any(item.get("qualifier_relation") == "conflicts" for item, _ in rows)
    contextual = (context_subclaim and cited_qualifier
                  and any(item.get("stance") in {"supports", "mixed"} for item, _ in rows))
    grounded = bool(reliable and (cited_conflict or contextual))
    return {
        "grounded": grounded,
        "material_sources": len(publishers),
        "reliable_sources": len(reliable),
        "material_context": contextual,
        "contradictory_passages": cited_conflict,
    }
