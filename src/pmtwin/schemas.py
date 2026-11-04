"""API request and response models (contract: KYROS ROADMAP, API contract v1)."""

from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

SETTING_NAMES = frozenset({"setting1", "setting2", "setting3"})
SENSOR_NAMES = frozenset(f"s{number}" for number in range(1, 22))

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


def _require_exact_keys(values: Mapping[str, float], expected: frozenset[str]) -> None:
    missing, extra = expected - values.keys(), values.keys() - expected
    if missing or extra:
        raise ValueError(f"missing {sorted(missing)}, unexpected {sorted(extra)}")


class SensorReading(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine_id: int = Field(ge=1)
    cycle: int = Field(ge=1)
    settings: dict[str, FiniteFloat]
    sensors: dict[str, FiniteFloat]

    @field_validator("settings")
    @classmethod
    def _settings_are_complete(cls, values: dict[str, float]) -> dict[str, float]:
        _require_exact_keys(values, SETTING_NAMES)
        return values

    @field_validator("sensors")
    @classmethod
    def _sensors_are_complete(cls, values: dict[str, float]) -> dict[str, float]:
        _require_exact_keys(values, SENSOR_NAMES)
        return values


class IngestRequest(BaseModel):
    readings: list[SensorReading] = Field(min_length=1, max_length=1000)


class Status(StrEnum):
    HEALTHY = "HEALTHY"
    WATCH = "WATCH"
    CRITICAL = "CRITICAL"


class RulEstimate(BaseModel):
    value: float
    model: str


class TwinSnapshot(BaseModel):
    engine_id: int
    life: int
    cycle: int
    rul: RulEstimate
    status: Status
    recommendation: str
    updated_at: datetime


class IngestResponse(BaseModel):
    snapshots: list[TwinSnapshot]


class FleetItem(BaseModel):
    engine_id: int
    life: int
    cycle: int
    status: Status
    rul: float


class HistoryPoint(BaseModel):
    cycle: int
    rul: float
    status: Status
    sensors: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    models: dict[str, str]
