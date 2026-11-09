"""Health model: how far a reading is from normal, and which sensors cause it.

Normal is learned with PCA on early-life readings (KYROS ROADMAP, ML protocol). Two distances:
T² is distance inside the normal pattern, SPE is the part the pattern cannot explain.
The notebook fits it; the twin and the notebook score with this same code.
"""

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HealthModel:
    sensors: list[str]
    scaler: Any  # fitted sklearn StandardScaler
    pca: Any  # fitted sklearn PCA on the scaled healthy readings
    t2_limit: float = math.inf
    spe_limit: float = math.inf

    def with_limits(self, t2_limit: float, spe_limit: float) -> HealthModel:
        return replace(self, t2_limit=float(t2_limit), spe_limit=float(spe_limit))

    def _residuals(self, readings: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        scaled = self.scaler.transform(readings[self.sensors])
        patterns = self.pca.transform(scaled)
        return patterns, scaled - self.pca.inverse_transform(patterns)

    def scores(self, readings: pd.DataFrame) -> pd.DataFrame:
        """Per reading: `t2`, `spe`, `score` (1.0 = on the alarm line) and `over`."""
        patterns, residual = self._residuals(readings)
        t2 = (patterns**2 / self.pca.explained_variance_).sum(axis=1)
        spe = (residual**2).sum(axis=1)
        score = np.maximum(t2 / self.t2_limit, spe / self.spe_limit)
        return pd.DataFrame(
            {"t2": t2, "spe": spe, "score": score, "over": score > 1.0}, index=readings.index
        )

    def top_sensors(self, readings: pd.DataFrame, k: int = 3) -> list[list[str]]:
        """Per reading, the k sensors furthest from their healthy mean, in units of their spread.

        Not SPE contributions: 99% of FD001 validation alarm cycles are T²-driven, and SPE then
        names sensors that barely moved (KYROS ADR-008).
        """
        scaled = self.scaler.transform(readings[self.sensors])
        order = np.argsort(-np.abs(scaled), axis=1)[:, :k]
        return [[self.sensors[index] for index in row] for row in order]


def persistent_alarm(over: pd.Series, need: int = 3, of: int = 5) -> pd.Series:
    """True once at least `need` of the last `of` readings were over a limit."""
    counts = over.astype(int).rolling(of, min_periods=1).sum()
    return pd.Series(np.asarray(counts) >= need, index=over.index)
