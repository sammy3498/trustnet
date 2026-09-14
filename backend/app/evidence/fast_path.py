import logging
import math
import re
import threading
from difflib import SequenceMatcher
from functools import lru_cache

import numpy as np
from sentence_transformers import CrossEncoder

from backend.app.evidence.domain import (
    get_publisher_domain,
)
from backend.app.evidence.models import (
    RetrievedEvidence,
)
from backend.app.retrieval.excerpt import (
    get_relevant_excerpt,
)


logger = logging.getLogger(
    __name__
)


FAST_PATH_MODEL_NAME = (
    "cross-encoder/nli-deberta-v3-base"
)

FAST_PATH_MAX_CLAIM_WORDS = 14
FAST_PATH_CONFIDENCE_THRESHOLD = 0.95
FAST_PATH_OPPOSITION_THRESHOLD = 0.80
FAST_PATH_DUPLICATE_SIMILARITY = 0.92
FAST_PATH_MAX_SENTENCES_PER_SOURCE = 8
FAST_PATH_MIN_INDEPENDENT_EVIDENCE = 2


RELIABLE_QUALITIES = {
    "high",
    "medium",
}


QUALITY_PRIORITY = {
    "high": 0,
    "medium": 1,
    "low": 2,
}


COMPLEX_CLAIM_MARKERS = (
    " and ",
    " or ",
    " because ",
    " while ",
    " whereas ",
    " although ",
    " but ",
    " unless ",
    " if ",
    " compared ",
    " compared with ",
    " compared to ",
    " than ",
)


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


_MODEL_INFERENCE_LOCK = (
    threading.Lock()
)


def normalize_text(
    text: str,
) -> str:
    value = text.lower()

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    value = re.sub(
        r"[^\w\s]",
        "",
        value,
    )

    return value.strip()


def normalize_content_term(
    word: str,
) -> str:
    value = word.lower()

    if (
        len(value) > 4
        and value.endswith(
            "ies"
        )
    ):
        return (
            value[:-3]
            + "y"
        )

    if (
        len(value) > 4
        and value.endswith(
            "s"
        )
        and not value.endswith(
            (
                "ss",
                "us",
                "is",
            )
        )
    ):
        return value[:-1]

    return value


def get_content_terms(
    text: str,
) -> set[str]:
    words = re.findall(
        r"[A-Za-z]+",
        text.lower(),
    )

    terms = set()

    for word in words:
        if (
            len(word) < 3
            or word in STOPWORDS
        ):
            continue

        terms.add(
            normalize_content_term(
                word
            )
        )

    return terms


def has_enough_term_overlap(
    claim: str,
    sentence: str,
) -> bool:
    claim_terms = (
        get_content_terms(
            claim
        )
    )

    sentence_terms = (
        get_content_terms(
            sentence
        )
    )

    if not claim_terms:
        return False

    overlap = (
        claim_terms
        & sentence_terms
    )

    if (
        len(
            claim_terms
        )
        <= 3
    ):
        required = len(
            claim_terms
        )

    else:
        required = max(
            3,
            math.ceil(
                len(
                    claim_terms
                )
                * 0.75
            ),
        )

    return (
        len(
            overlap
        )
        >= required
    )


def is_simple_atomic_claim(
    claim: str,
) -> bool:
    normalized = (
        " "
        + claim.strip().lower()
        + " "
    )

    if not claim.strip():
        return False

    if (
        len(
            claim.split()
        )
        > FAST_PATH_MAX_CLAIM_WORDS
    ):
        return False

    if re.search(
        r"\d",
        claim,
    ):
        return False

    if any(
        symbol in claim
        for symbol in (
            "%",
            "$",
            "€",
            "₹",
            ";",
            ":",
            "\n",
        )
    ):
        return False

    for marker in (
        COMPLEX_CLAIM_MARKERS
    ):
        if marker in normalized:
            return False

    return True


def split_sentences(
    text: str,
) -> list[str]:
    parts = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    sentences = []

    for part in parts:
        sentence = (
            part.strip()
        )

        if len(
            sentence
        ) < 20:
            continue

        sentences.append(
            sentence
        )

    return sentences


@lru_cache(
    maxsize=1
)
def get_fast_path_model():
    logger.warning(
        "Loading TrustNet local "
        "NLI fast-path model."
    )

    return CrossEncoder(
        FAST_PATH_MODEL_NAME,
        local_files_only=True,
    )


def softmax(
    values,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = (
        values
        - np.max(
            values,
            axis=1,
            keepdims=True,
        )
    )

    exp_values = np.exp(
        values
    )

    return (
        exp_values
        / np.sum(
            exp_values,
            axis=1,
            keepdims=True,
        )
    )


def get_nli_label_order(
    model,
) -> list[str]:
    default = [
        "contradiction",
        "entailment",
        "neutral",
    ]

    try:
        config = (
            model.model.config
        )

        id_to_label = (
            config.id2label
        )

        labels = []

        for index in range(3):
            labels.append(
                str(
                    id_to_label[
                        index
                    ]
                ).lower()
            )

        required = {
            "contradiction",
            "entailment",
            "neutral",
        }

        if set(
            labels
        ) == required:
            return labels

    except Exception:
        pass

    return default


def score_evidence_sentences(
    claim: str,
    evidence: list[
        RetrievedEvidence
    ],
) -> list[dict]:
    pairs = []
    metadata = []

    for evidence_index, item in (
        enumerate(
            evidence
        )
    ):
        excerpt = (
            get_relevant_excerpt(
                text=item.text,
                query=claim,
                max_chars=1200,
            )
        )

        sentences = (
            split_sentences(
                excerpt
            )
        )

        relevant_sentences = [
            sentence
            for sentence
            in sentences
            if has_enough_term_overlap(
                claim,
                sentence,
            )
        ]

        relevant_sentences = (
            relevant_sentences[
                :FAST_PATH_MAX_SENTENCES_PER_SOURCE
            ]
        )

        for sentence in (
            relevant_sentences
        ):
            pairs.append(
                (
                    sentence,
                    claim,
                )
            )

            metadata.append(
                {
                    "evidence_index": (
                        evidence_index
                    ),
                    "sentence": (
                        sentence
                    ),
                }
            )

    if not pairs:
        return []

    try:
        model = (
            get_fast_path_model()
        )

        with _MODEL_INFERENCE_LOCK:
            raw_scores = (
                model.predict(
                    pairs
                )
            )

        probabilities = (
            softmax(
                raw_scores
            )
        )

        label_order = (
            get_nli_label_order(
                model
            )
        )

    except Exception as exc:
        logger.warning(
            "Local NLI fast path "
            "is unavailable: %s",
            exc,
        )

        return []

    best_by_evidence = {}

    for (
        item_metadata,
        probability_row,
    ) in zip(
        metadata,
        probabilities,
    ):
        scores = {
            label: float(
                probability_row[
                    index
                ]
            )
            for index, label
            in enumerate(
                label_order
            )
        }

        entailment = (
            scores.get(
                "entailment",
                0.0,
            )
        )

        contradiction = (
            scores.get(
                "contradiction",
                0.0,
            )
        )

        neutral = (
            scores.get(
                "neutral",
                0.0,
            )
        )

        winner = max(
            scores,
            key=scores.get,
        )

        if winner == "entailment":
            stance = "supports"
            confidence = entailment

        elif winner == "contradiction":
            stance = "contradicts"
            confidence = contradiction

        else:
            stance = "neutral"
            confidence = neutral

        evidence_index = (
            item_metadata[
                "evidence_index"
            ]
        )

        candidate = {
            "evidence_index": (
                evidence_index
            ),
            "sentence": (
                item_metadata[
                    "sentence"
                ]
            ),
            "stance": stance,
            "confidence": confidence,
            "entailment": entailment,
            "contradiction": contradiction,
            "neutral": neutral,
        }

        current = (
            best_by_evidence.get(
                evidence_index
            )
        )

        candidate_decisiveness = max(
            entailment,
            contradiction,
        )

        if current is None:
            best_by_evidence[
                evidence_index
            ] = candidate
            continue

        current_decisiveness = max(
            current[
                "entailment"
            ],
            current[
                "contradiction"
            ],
        )

        if (
            candidate_decisiveness
            > current_decisiveness
        ):
            best_by_evidence[
                evidence_index
            ] = candidate

    return [
        best_by_evidence[
            index
        ]
        for index in sorted(
            best_by_evidence
        )
    ]


def sentences_are_duplicates(
    first: str,
    second: str,
) -> bool:
    first_normalized = (
        normalize_text(
            first
        )
    )

    second_normalized = (
        normalize_text(
            second
        )
    )

    if (
        not first_normalized
        or not second_normalized
    ):
        return False

    similarity = (
        SequenceMatcher(
            None,
            first_normalized,
            second_normalized,
        ).ratio()
    )

    return (
        similarity
        >= FAST_PATH_DUPLICATE_SIMILARITY
    )


def choose_cluster_representative(
    indexes: list[int],
    evidence: list[
        RetrievedEvidence
    ],
    matches: list[dict],
) -> int:
    return min(
        indexes,
        key=lambda index: (
            QUALITY_PRIORITY.get(
                evidence[
                    matches[
                        index
                    ][
                        "evidence_index"
                    ]
                ].quality,
                3,
            ),
            -matches[
                index
            ][
                "confidence"
            ],
            evidence[
                matches[
                    index
                ][
                    "evidence_index"
                ]
            ].url,
        ),
    )


def cluster_decisive_matches(
    evidence: list[
        RetrievedEvidence
    ],
    matches: list[dict],
) -> tuple[
    list[list[int]],
    list[int],
]:
    decisive_indexes = [
        index
        for index, match
        in enumerate(
            matches
        )
        if (
            match[
                "stance"
            ]
            in {
                "supports",
                "contradicts",
            }
            and match[
                "confidence"
            ]
            >= FAST_PATH_CONFIDENCE_THRESHOLD
        )
    ]

    clusters = []

    for index in (
        decisive_indexes
    ):
        sentence = (
            matches[
                index
            ][
                "sentence"
            ]
        )

        placed = False

        for cluster in clusters:
            representative_index = (
                cluster[
                    0
                ]
            )

            representative_sentence = (
                matches[
                    representative_index
                ][
                    "sentence"
                ]
            )

            if sentences_are_duplicates(
                sentence,
                representative_sentence,
            ):
                cluster.append(
                    index
                )

                placed = True
                break

        if not placed:
            clusters.append(
                [
                    index
                ]
            )

    representatives = [
        choose_cluster_representative(
            indexes=cluster,
            evidence=evidence,
            matches=matches,
        )
        for cluster in clusters
    ]

    return (
        clusters,
        representatives,
    )


def build_fast_path_analysis(
    claim: str,
    evidence: list[
        RetrievedEvidence
    ],
    matches: list[dict],
) -> dict | None:
    if not matches:
        return None

    (
        _,
        representatives,
    ) = cluster_decisive_matches(
        evidence=evidence,
        matches=matches,
    )

    reliable_representatives = []

    representative_publishers = (
        set()
    )

    for representative_index in (
        representatives
    ):
        match = matches[
            representative_index
        ]

        item = evidence[
            match[
                "evidence_index"
            ]
        ]

        if (
            item.quality
            not in RELIABLE_QUALITIES
        ):
            continue

        publisher = (
            get_publisher_domain(
                item.source
                or item.url
            )
        )

        if not publisher:
            publisher = item.url

        if (
            publisher
            in representative_publishers
        ):
            continue

        representative_publishers.add(
            publisher
        )

        reliable_representatives.append(
            representative_index
        )

    if (
        len(
            reliable_representatives
        )
        < FAST_PATH_MIN_INDEPENDENT_EVIDENCE
    ):
        return None

    representative_stances = {
        matches[
            index
        ][
            "stance"
        ]
        for index
        in reliable_representatives
    }

    if (
        len(
            representative_stances
        )
        != 1
    ):
        return None

    direction = next(
        iter(
            representative_stances
        )
    )

    opposite_key = (
        "contradiction"
        if direction == "supports"
        else "entailment"
    )

    for match in matches:
        if (
            match[
                opposite_key
            ]
            >= FAST_PATH_OPPOSITION_THRESHOLD
        ):
            return None

    representative_match_indexes = (
        set(
            reliable_representatives
        )
    )

    evidence_assessments = []

    high_publishers = set()
    medium_publishers = set()

    effective_evidence_count = 0

    for match_index, match in (
        enumerate(
            matches
        )
    ):
        item = evidence[
            match[
                "evidence_index"
            ]
        ]

        is_representative = (
            match_index
            in representative_match_indexes
        )

        if (
            is_representative
            and match[
                "stance"
            ]
            == direction
        ):
            relevance = "high"
            directness = "direct"

            effective_evidence_count += 1

            publisher = (
                get_publisher_domain(
                    item.source
                    or item.url
                )
            )

            if not publisher:
                publisher = item.url

            if item.quality == "high":
                high_publishers.add(
                    publisher
                )

            elif (
                item.quality
                == "medium"
            ):
                medium_publishers.add(
                    publisher
                )

            reasoning = (
                "This source contains a "
                "direct statement that "
                f"{direction} the claim."
            )

        elif (
            match[
                "stance"
            ]
            == direction
            and match[
                "confidence"
            ]
            >= FAST_PATH_CONFIDENCE_THRESHOLD
        ):
            relevance = "low"
            directness = "partial"

            reasoning = (
                "This source also "
                f"{direction} the claim, "
                "but it was not counted as "
                "independent reliable "
                "corroboration."
            )

        else:
            relevance = "low"
            directness = "none"

            reasoning = (
                "This source was not counted "
                "as decisive independent "
                "evidence for the fast path."
            )

        evidence_assessments.append(
            {
                "title": (
                    item.title
                ),
                "url": item.url,
                "stance": (
                    match[
                        "stance"
                    ]
                ),
                "quality": (
                    item.quality
                ),
                "relevance": relevance,
                "directness": directness,
                "qualifier_relation": (
                    "not_applicable"
                ),
                "reasoning": reasoning,
            }
        )

    supports = (
        direction
        == "supports"
    )

    summary = (
        "Independent reliable evidence "
        "sentences consistently "
        f"{direction} the claim."
    )

    return {
        "claim": claim,
        "evidence_assessments": (
            evidence_assessments
        ),
        "subclaim_assessments": [
            {
                "subclaim": claim,
                "coverage": "complete",
                "direct_support": (
                    supports
                ),
                "direct_contradiction": (
                    not supports
                ),
                "material_context_conflict": (
                    False
                ),
                "status": direction,
                "reasoning": summary,
            }
        ],
        "overall_evidence_direction": (
            direction
        ),
        "summary": summary,
        "bias_framing": "neutral",
        "bias_framing_justification": (
            "No material framing concern "
            "was established for this "
            "simple factual claim."
        ),
        "incentive_conflict": (
            "none_detected"
        ),
        "incentive_conflict_justification": (
            "No incentive or conflict "
            "was established from the "
            "supplied evidence."
        ),
        "material_caveats": [],
        "_fast_path": {
            "route": "local_nli",
            "effective_evidence_count": (
                effective_evidence_count
            ),
            "effective_high_quality_count": (
                len(
                    high_publishers
                )
            ),
            "effective_medium_quality_count": (
                len(
                    medium_publishers
                )
            ),
        },
    }


def try_fast_evidence_analysis(
    claim: str,
    evidence: list[
        RetrievedEvidence
    ],
) -> dict | None:
    if not is_simple_atomic_claim(
        claim
    ):
        return None

    matches = (
        score_evidence_sentences(
            claim=claim,
            evidence=evidence,
        )
    )

    return build_fast_path_analysis(
        claim=claim,
        evidence=evidence,
        matches=matches,
    )