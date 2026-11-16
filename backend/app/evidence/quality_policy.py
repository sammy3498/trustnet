from backend.app.evidence.conflict_verdict import grounded_mixed_evidence
"""Rules for accepting source-grounded directional findings.

Quality describes publisher trust; *material relevance* must be established
separately.  Requiring two corroborations for every refutation would make a
single authoritative counterexample incapable of disproving "never" claims.
"""
from backend.app.evidence.domain import get_publisher_domain


def prefer_comparable_sources(selected, ranked, query_ranks, publisher):
    """Retained as a compatibility helper; final selection is facet-aware."""
    out = list(selected)
    positions = {doc.url: i for i, doc in enumerate(ranked)}
    tier = {"high": 2, "medium": 1, "low": 0}
    for i, old in enumerate(out):
        if old.quality != "low" or old.url not in positions:
            continue
        used = {publisher(doc.source or doc.url) for j, doc in enumerate(out) if j != i}
        candidates = [doc for doc in ranked if doc.url not in {d.url for d in out}
                      and tier.get(doc.quality, 0) > 0
                      and doc.semantic_score >= old.semantic_score - 0.04
                      and doc.bm25_score >= old.bm25_score * 0.85
                      and doc.reranker_score >= old.reranker_score
                      and publisher(doc.source or doc.url) not in used]
        if candidates:
            out[i] = max(candidates, key=lambda d: (tier.get(d.quality, 0), -positions.get(d.url, 100)))
    return out


def relevant_assessments(assessments, evidence, direction=None, allow_partial=False):
    by_url = {doc.url.rstrip("/"): doc for doc in evidence}
    aligned = []
    for row in assessments:
        stance = row.get("stance")
        if stance not in {"supports", "contradicts", "mixed"}:
            continue
        if direction and stance not in {direction, "mixed"}:
            continue
        if row.get("relevance") not in {"high", "medium"}:
            continue
        if row.get("directness") not in ({"direct", "partial"} if allow_partial else {"direct"}):
            continue
        doc = by_url.get(str(row.get("url", "")).rstrip("/"))
        if doc:
            aligned.append((row, doc))
    return aligned


def directional_gate(direction, assessments, evidence, publisher, medium_needed, subclaims=None):
    if direction == "mixed":
        metadata = grounded_mixed_evidence({
            "evidence_assessments": assessments,
            "subclaim_assessments": subclaims or [],
        }, evidence)
        if metadata["grounded"]:
            return None
        return "No sufficiently authoritative, directly cited material conflict was established."
    if direction not in {"supports", "contradicts"}:
        return "No settled source-grounded direction was established."
    # Partial documents can jointly establish a compound support claim, but
    # a decisive refutation must identify a direct counterexample.
    allow_partial = direction == "supports" and len(subclaims or []) > 1
    rows = relevant_assessments(assessments, evidence, direction, allow_partial)
    opposing = relevant_assessments(
        assessments, evidence,
        "contradicts" if direction == "supports" else "supports",
        allow_partial=False,
    )
    statuses = {sub.get("status") for sub in subclaims or []}
    different_conjuncts = (direction == "contradicts"
                           and len(subclaims or []) > 1
                           and "supports" in statuses and "contradicts" in statuses
                           and "mixed" not in statuses)
    if opposing and not different_conjuncts:
        return "Reliable direct evidence on both sides remains unresolved."
    if not rows:
        return "No cited passage directly establishes the claim's verdict."
    strengths = {}
    for row, doc in rows:
        key = publisher(doc.source or doc.url)
        quality = {"low": 0, "medium": 1, "high": 2}.get(doc.quality, 0)
        strengths[key] = max(strengths.get(key, -1), quality)
    if 2 in strengths.values() or 1 in strengths.values():
        return None
    if len(strengths) >= 2:
        return None  # Two independent, direct low-tier passages: cautious verdict
    return "Only one low-authority source directly addresses the claim. Independent confirmation is needed."
