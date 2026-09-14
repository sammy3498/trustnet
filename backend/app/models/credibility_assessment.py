from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
)


BiasFramingAssessment = Literal[
    "neutral",
    "minor_concern",
    "material_concern",
    "severe_concern",
]


IncentiveConflictAssessment = Literal[
    "none_detected",
    "minor_concern",
    "material_concern",
    "severe_concern",
]


class CredibilityAssessmentResponse(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    claim: str

    bias_framing: (
        BiasFramingAssessment
    )

    bias_framing_justification: str

    incentive_conflict: (
        IncentiveConflictAssessment
    )

    incentive_conflict_justification: str

    material_caveats: list[str]