from backend.app.models.credibility_assessment import (
    CredibilityAssessmentResponse,
)


def test_strict_credibility_schema_requires_every_property():
    schema = (
        CredibilityAssessmentResponse
        .model_json_schema()
    )

    properties = set(
        schema[
            "properties"
        ].keys()
    )

    required = set(
        schema[
            "required"
        ]
    )

    assert required == properties

    assert (
        "material_caveats"
        in required
    )