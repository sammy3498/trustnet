import argparse
import json
from pathlib import Path

from backend.app.evidence.models import RetrievedEvidence
from backend.app.llm.evidence_analysis import (
    analyze_evidence_with_llm,
    load_evidence_prompt,
)
from backend.app.retrieval.excerpt import (
    DEFAULT_PASSAGE_MAX_CHARS,
    get_relevant_excerpt,
)


INPUT_PATH = Path(
    "eval/results/retrieval_v2_2_1998_2674.json"
)

OUTPUT_ROOT = Path(
    "eval/results"
)

LABEL_MAP = {
    "supports": "Supported",
    "contradicts": "Refuted",
    "mixed": "Conflicting Evidence/Cherrypicking",
    "insufficient": "Not Enough Evidence",
}


def load_cases() -> list[dict]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input: {INPUT_PATH}"
        )

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def get_case(
    index: int,
) -> dict:
    cases = load_cases()

    for case in cases:
        if case.get("index") == index:
            return case

    raise ValueError(
        f"Index {index} not found in {INPUT_PATH}"
    )


def build_evidence(
    case: dict,
) -> list[RetrievedEvidence]:
    return [
        RetrievedEvidence.model_validate(item)
        for item in case.get(
            "new_evidence",
            [],
        )
    ]


def estimate_prompt_chars(
    claim: str,
    evidence: list[RetrievedEvidence],
) -> int:
    system_prompt = load_evidence_prompt()

    evidence_parts = []

    for item in evidence:
        excerpt = get_relevant_excerpt(
            text=item.text,
            query=claim,
            max_chars=DEFAULT_PASSAGE_MAX_CHARS,
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

    evidence_text = "\n\n".join(
        evidence_parts
    )

    user_prompt = f"""
Claim:
{claim}

Evidence:
{evidence_text}
"""

    return (
        len(system_prompt)
        + len(user_prompt)
    )


def get_output_path(
    index: int,
) -> Path:
    return (
        OUTPUT_ROOT
        / (
            "retrieval_v2_2_"
            f"evidence_analysis_{index}.json"
        )
    )


def get_raw_output_path(
    index: int,
) -> Path:
    return (
        OUTPUT_ROOT
        / (
            "retrieval_v2_2_"
            f"evidence_analysis_{index}_raw.json"
        )
    )


def save_json(
    path: Path,
    data,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temporary_path.replace(path)


def process_analysis(
    *,
    index: int,
    claim: str,
    gold_label: str,
    evidence: list[RetrievedEvidence],
    analysis: dict,
    output_path: Path,
):
    direction = analysis.get(
        "overall_evidence_direction"
    )

    if direction is None:
        raise KeyError(
            "Evidence Analysis result does not "
            "contain 'overall_evidence_direction'. "
            f"Available keys: {sorted(analysis.keys())}"
        )

    if direction not in LABEL_MAP:
        raise ValueError(
            "Unexpected evidence direction: "
            f"{direction}"
        )

    predicted_label = LABEL_MAP[
        direction
    ]

    result = {
        "index": index,
        "claim": claim,
        "gold_label": gold_label,
        "predicted_label": predicted_label,
        "correct": (
            predicted_label
            == gold_label
        ),
        "evidence_direction": direction,
        "llm_evidence_direction": (
            analysis.get(
                "llm_overall_evidence_direction"
            )
        ),
        "direction_was_resolved": (
            analysis.get(
                "direction_was_resolved",
                False,
            )
        ),
        "analysis": analysis,
        "evidence": [
            item.model_dump()
            for item in evidence
        ],
    }

    save_json(
        output_path,
        result,
    )

    print(
        "PREDICTED:",
        predicted_label,
    )
    print(
        "CORRECT:",
        result["correct"],
    )
    print(
        "DIRECTION:",
        direction,
    )
    print(
        "LLM DIRECTION:",
        result[
            "llm_evidence_direction"
        ],
    )
    print(
        "DIRECTION RESOLVED:",
        result[
            "direction_was_resolved"
        ],
    )
    print(
        "SAVED:",
        output_path,
    )


def run(
    index: int,
    dry_run: bool,
    max_attempts: int,
):
    case = get_case(index)

    claim = case["claim"]
    gold_label = case["gold_label"]

    evidence = build_evidence(
        case
    )

    prompt_chars = estimate_prompt_chars(
        claim=claim,
        evidence=evidence,
    )

    output_path = get_output_path(
        index
    )

    raw_output_path = (
        get_raw_output_path(
            index
        )
    )

    print(
        "===== CACHED RETRIEVAL "
        "V2.2 ANALYSIS ====="
    )
    print(
        "INDEX:",
        index,
    )
    print(
        "GOLD:",
        gold_label,
    )
    print(
        "EVIDENCE ITEMS:",
        len(evidence),
    )
    print(
        "PROMPT CHARS:",
        prompt_chars,
    )
    print(
        "APPROX INPUT TOKENS:",
        round(prompt_chars / 4),
    )

    if dry_run:
        print(
            "OUTPUT EXISTS:",
            output_path.exists(),
        )
        print(
            "RAW OUTPUT EXISTS:",
            raw_output_path.exists(),
        )
        print("OPENAI CALLS: 0")
        print("TAVILY CALLS: 0")
        return

    if output_path.exists():
        raise FileExistsError(
            "Completed output already exists. "
            "Refusing another paid call: "
            f"{output_path}"
        )

    if raw_output_path.exists():
        print(
            "RAW RESULT FOUND. "
            "RECOVERING WITHOUT API CALL."
        )

        with raw_output_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            analysis = json.load(
                file
            )

        process_analysis(
            index=index,
            claim=claim,
            gold_label=gold_label,
            evidence=evidence,
            analysis=analysis,
            output_path=output_path,
        )

        print("OPENAI CALLS: 0")
        print("TAVILY CALLS: 0")
        return

    if max_attempts != 1:
        raise ValueError(
            "For a new paid experiment, "
            "--max-attempts must be exactly 1."
        )

    if not evidence:
        raise ValueError(
            "No cached evidence available."
        )

    analysis = analyze_evidence_with_llm(
        claim=claim,
        evidence=evidence,
    )

    if not isinstance(
        analysis,
        dict,
    ):
        raise TypeError(
            "Evidence Analysis returned "
            f"{type(analysis).__name__}, "
            "expected dict."
        )

    # Save the successful API result immediately.
    # Nothing after this point can cause the
    # paid response itself to be lost.
    save_json(
        raw_output_path,
        analysis,
    )

    print(
        "RAW RESULT SAVED:",
        raw_output_path,
    )
    print(
        "RAW KEYS:",
        sorted(
            analysis.keys()
        ),
    )

    process_analysis(
        index=index,
        claim=claim,
        gold_label=gold_label,
        evidence=evidence,
        analysis=analysis,
        output_path=output_path,
    )

    print("OPENAI CALLS: 1")
    print("TAVILY CALLS: 0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--index",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    run(
        index=args.index,
        dry_run=args.dry_run,
        max_attempts=args.max_attempts,
    )