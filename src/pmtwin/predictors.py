"""Models behind one small interface, so the twin and its tests do not depend on a library."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, Self

import joblib
import numpy as np
import pandas as pd

from pmtwin.features import add_rolling_means

Window = Sequence[Mapping[str, float]]


class RulPredictor(Protocol):
    name: str
    window: int

    def predict(self, windows: Sequence[Window]) -> list[float]:
        """One RUL per window; each window is an engine's recent readings, oldest first."""
        ...


class GbmPredictor:
    name = "lgbm"

    def __init__(self, model: Any, features: list[str], window: int) -> None:
        self.model = model
        self.features = features
        self.window = window
        self._sensors = [name for name in features if not name.endswith("_mean")]

    @classmethod
    def load(cls, path: Path) -> Self:
        artifact = joblib.load(path)
        return cls(artifact["model"], artifact["features"], artifact["window"])

    def predict(self, windows: Sequence[Window]) -> list[float]:
        rows = [{"unit": index, **row} for index, window in enumerate(windows) for row in window]
        featured = add_rolling_means(pd.DataFrame(rows), self._sensors, self.window)
        latest = featured.groupby("unit").tail(1)
        return np.clip(self.model.predict(latest[self.features]), 0, None).tolist()
