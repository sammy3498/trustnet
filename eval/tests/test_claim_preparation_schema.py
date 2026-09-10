import pytest
from pydantic import (
    ValidationError,
)

from backend.app.models.claim_analysis import (
    ClaimPreparationResponse,
)


def base_response() -> dict:
    return {
        "normalized_claim": (
            "Test claim."
        ),
        "is_clear": True,
        "reason": (
            "The claim is verifiable."
        ),
        "search_query": (
            "test claim evidence"
        ),
        "verification_queries": [
            "test claim evidence"
        ],
    }


def test_valid_claim_preparation():
    data = base_response()

    result = (
        ClaimPreparationResponse
        .model_validate(
            data
        )
    )

    assert (
        result.normalized_claim
        == "Test claim."
    )

    assert len(
        result.verification_queries
    ) == 1


def test_missing_verification_queries_rejected():
    data = base_response()

    del data[
        "verification_queries"
    ]

    with pytest.raises(
        ValidationError
    ):
        ClaimPreparationResponse.model_validate(
            data
        )


def test_extra_field_rejected():
    data = base_response()

    data[
        "unexpected_field"
    ] = "unexpected"

    with pytest.raises(
        ValidationError
    ):
        ClaimPreparationResponse.model_validate(
            data
        )


def test_unclear_claim_can_have_empty_queries():
    data = base_response()

    data[
        "is_clear"
    ] = False

    data[
        "verification_queries"
    ] = []

    result = (
        ClaimPreparationResponse
        .model_validate(
            data
        )
    )

    assert (
        result.verification_queries
        == []
    )


def test_json_schema_is_strict_ready():
    schema = (
        ClaimPreparationResponse
        .model_json_schema()
    )

    assert (
        schema[
            "additionalProperties"
        ]
        is False
    )

    required = set(
        schema[
            "required"
        ]
    )

    assert required == {
        "normalized_claim",
        "is_clear",
        "reason",
        "search_query",
        "verification_queries",
    }