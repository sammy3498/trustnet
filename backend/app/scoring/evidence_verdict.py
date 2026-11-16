"""Direction-aware score reconciliation, with evidence as the deciding factor.

The five factor scores remain available for diagnostic interpretation.  A high
bias/incentive factor must not outweigh documented factual contradiction.
"""

def reconcile(direction, factors, weighted=None, label_for_score=None):
    if direction == "mixed":
        # Misleading/cherry-picked context is not a calibrated truth percentage.
        # The UI should show an explicit verdict and an unavailable score.
        return None, None
    if direction not in {"supports", "contradicts"}:
        return None, "The material evidence does not establish a settled factual direction."
    evidence_score = max(0, min(100, int(factors["evidence_strength"]["score"])))
    factual = factors.get("factual_consistency", {})
    original = factual.get("score")
    # Separate factual direction from evidence strength/quality.  The overall
    # score is evidence-grounded and reflects support *and* reliability.
    resolved_factual = 100 if direction == "supports" else 0
    if original != resolved_factual:
        factors["factual_consistency"] = {
            "score": resolved_factual,
            "justification": (
                f"The separate LLM factor estimate ({original}/100) did not "
                "consistently describe the source-grounded material subclaim "
                f"verdict ({direction}). The directional fact assessment takes "
                "precedence. Source assessment: " +
                factors["evidence_strength"].get("justification", "")
            ),
        }
    # The 40/35/10/8/7 weighted score is retained as a diagnostic formula
    # but intentionally not used as a truth score.  Otherwise unrelated
    # framing/incentive scores can flip a verified falsehood into credibility.
    provenance = int(factors.get("source_provenance", {}).get("score", 0))
    if direction == "supports":
        # A semantic finding of support must not be labelled "Low Credibility"
        # just because the retrieval engine found only weaker publishers.  This
        # score is an ordinal heuristic, not a calibrated probability of truth.
        # Keep high-quality evidence's existing score; bound weaker support to
        # an explicitly tentative/moderate band while confidence stays lower.
        score = max(70, min(79, evidence_score)) if provenance < 60 else max(75, min(100, evidence_score))
    else:
        score = max(0, min(49, evidence_score))
    return score, None
