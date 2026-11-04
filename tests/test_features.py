"""Rolling features must stay inside one engine, because the twin computes them the same way."""

import pandas as pd

from pmtwin.features import add_rolling_means


def test_rolling_mean_uses_the_last_rows_of_the_same_engine_only():
    engines = pd.DataFrame({"unit": [1, 1, 1, 2], "s2": [1.0, 2.0, 4.0, 10.0]})

    result = add_rolling_means(engines, ["s2"], window=2)

    assert result["s2_mean"].tolist() == [1.0, 1.5, 3.0, 10.0]


def test_rolling_means_leave_the_input_unchanged():
    engines = pd.DataFrame({"unit": [1, 1], "s2": [1.0, 2.0]})

    add_rolling_means(engines, ["s2"], window=2)

    assert list(engines.columns) == ["unit", "s2"]
