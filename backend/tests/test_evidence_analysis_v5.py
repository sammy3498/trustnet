from backend.app.models.evidence_analysis import (
    EvidenceAnalysisResponseV5,
)


def test_v5_schema_is_strictly_required():
    schema = (
        EvidenceAnalysisResponseV5
        .model_json_schema()
    )

    properties = set(
        schema[
            "properties"
        ]
    )

    required = set(
        schema[
            "required"
        ]
    )

    assert required == properties

    assert (
        "bias_framing"
        in required
    )

    assert (
        "incentive_conflict"
        in required
    )

    assert (
        "material_caveats"
        in required
    )

    assert (
        "evidence_strength_score"
        not in properties
    )