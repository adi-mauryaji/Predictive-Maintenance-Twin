"""Features shared by the notebooks and the twin, so training and serving compute them alike."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def add_rolling_means(engines: pd.DataFrame, sensors: Sequence[str], window: int) -> pd.DataFrame:
    """Return a copy with `<sensor>_mean`: the mean of that engine's last `window` rows."""
    means = (
        engines.groupby("unit")[list(sensors)]
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level="unit", drop=True)
    )
    return engines.join(means.add_suffix("_mean"))


def engine_split(units: ArrayLike, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit (70), validation (15) and calibration (15) engines, the same on every run.

    Splitting by engine, never by row, keeps one engine's cycles out of two sets (leakage).
    """
    shuffled = np.random.default_rng(seed).permutation(np.asarray(units))
    return shuffled[:70], shuffled[70:85], shuffled[85:]
