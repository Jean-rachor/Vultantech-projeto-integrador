from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float]


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: list[dict[str, float]] = Field(min_length=1, max_length=500)


class ImageLabelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: Literal["NORMAL", "POROSIDADE", "TRINCA", "DESCONTINUIDADE"]
    label_source: str = Field(min_length=3, max_length=200)

    @field_validator("label_source")
    @classmethod
    def trim_label_source(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("Informe uma origem de rótulo com pelo menos três caracteres.")
        return value


class PredictResponse(BaseModel):
    diagnosis: dict[str, Any]
    probabilities: dict[str, float]
    decision: dict[str, Any]
