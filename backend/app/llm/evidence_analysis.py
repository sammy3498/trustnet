import json
from pathlib import Path

from pydantic import (
    ValidationError,
)

from backend.app.evidence.direction import (
    align_evidence_strength_score,
    resolve_overall_evidence_direction,
)
from backend.app.evidence.models import (
    RetrievedEvidence,
)
from backend.app.llm.client import (
    get_llm_response,
)
from backend.app.models.evidence_analysis import (
    EvidenceAnalysisResponseV4,
)
from backend.app.retrieval.excerpt import (
    DEFAULT_PASSAGE_MAX_CHARS,
    get_relevant_excerpt,
)


PROMPT_PATH = Path(
    "prompts/evidence_analysis_v4.txt"
)

EVIDENCE_ANALYSIS_SCHEMA = (
    EvidenceAnalysisResponseV4
    .model_json_schema()
)

NORMAL_EVIDENCE_MAX_CHARS = 1200


def load_evidence_prompt() -> str:
    return PROMPT_PATH.read_text(
        encoding="utf-8"
    )


def get_evidence_max_chars(
    mode: str,
) -> int:
    if mode == "normal":
        return (
            NORMAL_EVIDENCE_MAX_CHARS
        )

    return (
        DEFAULT_PASSAGE_MAX_CHARS
    )


def _normalized_url(
    value: str,
) -> str:
    return value.strip().rstrip("/")


def apply_retrieved_source_quality(
    analysis: dict,
    evidence: list[RetrievedEvidence],
) -> dict:
    """Bind every cited assessment to a retrieved item, never an LLM URL.

    Source rating and document identity are deterministic.  Unknown URLs are
    removed rather than allowing a plausible-sounding invented citation.
    """
    by_url = {_normalized_url(item.url): item for item in evidence}
    by_title = {}
    for item in evidence:
        by_title.setdefault(item.title.strip().lower(), []).append(item)
    clean = []
    omitted = 0
    seen = set()
    for row in analysis.get("evidence_assessments", []):
        key = _normalized_url(str(row.get("url", "")))
        doc = by_url.get(key)
        if doc is None:
            matches = by_title.get(str(row.get("title", "")).strip().lower(), [])
            if len(matches) == 1:
                doc = matches[0]
        if doc is None or doc.url in seen:
            omitted += 1
            continue
        seen.add(doc.url)
        row["url"] = doc.url
        row["title"] = doc.title
        row["quality"] = doc.quality
        row["evidence_origin"] = doc.evidence_origin
        if row.get("directness") == "none":
            row["stance"] = "neutral"
        clean.append(row)
    analysis["evidence_assessments"] = clean
    analysis["unmatched_citation_count"] = omitted
    return analysis


def enforce_subclaim_grounding(analysis: dict) -> dict:
    """A subclaim's 'direct' flag must have a matching cited source stance.

    This is a structural sanity check, not an independent semantic fact-check.
    """
    direct_support = any(
        row.get("stance") in {"supports", "mixed"}
        and row.get("directness") in {"direct", "partial"}
        and row.get("relevance") in {"high", "medium"}
        for row in analysis.get("evidence_assessments", [])
    )
    direct_contradiction = any(
        row.get("stance") in {"contradicts", "mixed"}
        and row.get("directness") == "direct"
        and row.get("relevance") in {"high", "medium"}
        for row in analysis.get("evidence_assessments", [])
    )
    for row in analysis.get("subclaim_assessments", []):
        if not direct_support:
            row["direct_support"] = False
        if not direct_contradiction:
            row["direct_contradiction"] = False
    return analysis


def apply_v4_subclaim_consistency(
    analysis: dict,
) -> dict:
    subclaims = analysis["subclaim_assessments"]
    # Material context must be attested by an actual retrieved passage;
    # a general model intuition is not enough to invent cherry-picking.
    grounded_qualifier = any(
        row.get("qualifier_relation") == "conflicts"
        and row.get("stance") in {"contradicts", "mixed"}
        and row.get("relevance") in {"high", "medium"}
        and row.get("directness") in {"direct", "partial"}
        for row in analysis.get("evidence_assessments", [])
    )

    for subclaim in subclaims:
        if subclaim.get("material_context_conflict") and not grounded_qualifier:
            subclaim["material_context_conflict"] = False
            subclaim["ungrounded_context_conflict_discarded"] = True
        original_status = (
            subclaim[
                "status"
            ]
        )

        subclaim[
            "llm_status"
        ] = original_status

        coverage = (
            subclaim[
                "coverage"
            ]
        )

        direct_support = bool(
            subclaim[
                "direct_support"
            ]
        )

        direct_contradiction = bool(
            subclaim[
                "direct_contradiction"
            ]
        )

        context_conflict = bool(
            subclaim[
                "material_context_conflict"
            ]
        )

        if (
            direct_support
            and direct_contradiction
        ):
            resolved_status = (
                "mixed"
            )

        elif (
            direct_support
            and context_conflict
        ):
            resolved_status = (
                "mixed"
            )

        elif direct_contradiction:
            resolved_status = (
                "contradicts"
            )

        elif (
            direct_support
            and coverage
            == "complete"
        ):
            resolved_status = (
                "supports"
            )

        else:
            resolved_status = (
                "insufficient"
            )

        subclaim[
            "status"
        ] = resolved_status

        subclaim[
            "status_was_resolved"
        ] = (
            resolved_status
            != original_status
        )

    return analysis


def apply_direction_consistency(
    analysis: dict,
) -> dict:
    original_direction = (
        analysis[
            "overall_evidence_direction"
        ]
    )

    original_score = (
        analysis[
            "evidence_strength_score"
        ]
    )

    subclaims = (
        analysis[
            "subclaim_assessments"
        ]
    )

    resolved_direction = (
        resolve_overall_evidence_direction(
            subclaim_assessments=(
                subclaims
            ),
            fallback_direction=(
                original_direction
            ),
            claim_text=analysis.get("claim", ""),
        )
    )

    resolved_score = (
        align_evidence_strength_score(
            score=original_score,
            original_direction=(
                original_direction
            ),
            resolved_direction=(
                resolved_direction
            ),
        )
    )

    analysis[
        "llm_overall_evidence_direction"
    ] = original_direction

    analysis[
        "llm_evidence_strength_score"
    ] = original_score

    analysis[
        "overall_evidence_direction"
    ] = resolved_direction

    analysis[
        "evidence_strength_score"
    ] = resolved_score

    analysis[
        "direction_was_resolved"
    ] = (
        resolved_direction
        != original_direction
    )

    return analysis


def analyze_evidence_with_llm(
    claim: str,
    evidence: list[
        RetrievedEvidence
    ],
    timeout_seconds: float | None = None,
    mode: str = "normal",
) -> dict:
    system_prompt = (
        load_evidence_prompt()
    )

    evidence_parts = []

    max_chars = (
        get_evidence_max_chars(
            mode
        )
    )

    for item in evidence:
        excerpt = (
            get_relevant_excerpt(
                text=item.text,
                query=(item.matched_query or claim),
                max_chars=max_chars,
            )
        )

        evidence_parts.append(
            f"""
Title: {item.title}
URL: {item.url}
Source: {item.source}
Quality: {item.quality}
Origin: {item.evidence_origin}
Evidence:
{excerpt}
"""
        )

    evidence_text = (
        "\n\n".join(
            evidence_parts
        )
    )

    user_prompt = f"""
Claim:
{claim}

Evidence:
{evidence_text}
"""

    raw_response = (
        get_llm_response(
            system_prompt=(
                system_prompt
            ),
            user_prompt=(
                user_prompt
            ),
            response_schema=(
                EVIDENCE_ANALYSIS_SCHEMA
            ),
            response_schema_name=(
                "trustnet_evidence_analysis_v4"
            ),
            timeout_seconds=(
                timeout_seconds
            ),
        )
    )

    try:
        data = json.loads(
            raw_response
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            "LLM returned invalid "
            f"JSON:\n{raw_response}"
        ) from exc

    try:
        validated = (
            EvidenceAnalysisResponseV4
            .model_validate(
                data
            )
        )

    except ValidationError as exc:
        raise ValueError(
            "LLM returned invalid "
            "evidence analysis:\n"
            f"{exc}"
        ) from exc

    analysis = (
        validated.model_dump()
    )

    analysis = (
        apply_retrieved_source_quality(
            analysis=analysis,
            evidence=evidence,
        )
    )

    analysis = enforce_subclaim_grounding(analysis)
    analysis = (
        apply_v4_subclaim_consistency(
            analysis
        )
    )

    return (
        apply_direction_consistency(
            analysis
        )
    )