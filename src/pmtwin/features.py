"""Features shared by the notebooks and the twin, so training and serving compute them alike."""

from collections.abc import Sequence

import pandas as pd


def add_rolling_means(engines: pd.DataFrame, sensors: Sequence[str], window: int) -> pd.DataFrame:
    """Return a copy with `<sensor>_mean`: the mean of that engine's last `window` rows."""
    means = (
        engines.groupby("unit")[list(sensors)]
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level="unit", drop=True)
    )
    return engines.join(means.add_suffix("_mean"))
