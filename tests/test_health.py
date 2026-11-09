"""Health model: distance from normal, which sensors cause it, and when it becomes an alarm."""

import math
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pmtwin.features import engine_split
from pmtwin.health import HealthModel, persistent_alarm


@pytest.fixture(scope="module")
def model():
    """a and b move together (one pattern), c moves alone (a second pattern)."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=500)
    normal = pd.DataFrame(
        {
            "a": base + 0.05 * rng.normal(size=500),
            "b": 2 * base + 0.1 * rng.normal(size=500),
            "c": rng.normal(size=500),
        }
    )
    scaler = StandardScaler().fit(normal)
    pca = PCA(n_components=0.95).fit(scaler.transform(normal))
    return HealthModel(["a", "b", "c"], scaler, pca)


def reading(a, b, c):
    return pd.DataFrame({"a": [a], "b": [b], "c": [c]})


def test_an_average_reading_is_close_to_normal(model):
    scores = model.scores(reading(0.0, 0.0, 0.0))

    assert scores["t2"].tolist()[0] < 0.1
    assert scores["spe"].tolist()[0] < 0.1


def test_extreme_but_consistent_reading_raises_t2_not_spe(model):
    scores = model.scores(reading(4.0, 8.0, 0.0))

    assert scores["t2"].tolist()[0] > 10
    assert scores["spe"].tolist()[0] < 0.5


def test_t2_measures_distance_in_units_of_normal_spread(model):
    """The a-b pattern varies about twice as much as c, so the same step along c is rarer."""
    step = 3.0
    along_ab = model.scaler.inverse_transform([[step / 2**0.5, step / 2**0.5, 0.0]])[0]
    along_c = model.scaler.inverse_transform([[0.0, 0.0, step]])[0]

    t2_ab = model.scores(reading(*along_ab))["t2"].tolist()[0]
    t2_c = model.scores(reading(*along_c))["t2"].tolist()[0]

    assert t2_c > 1.5 * t2_ab


def test_broken_relationship_raises_spe_and_names_its_sensors(model):
    broken = reading(3.0, -6.0, 0.0)

    scores = model.scores(broken)

    assert scores["spe"].tolist()[0] > 10
    assert set(model.top_sensors(broken, k=2)[0]) == {"a", "b"}


def test_limits_turn_scores_into_over_flags_and_a_ratio(model):
    limited = model.with_limits(t2_limit=10.0, spe_limit=5.0)

    scores = limited.scores(pd.concat([reading(0.0, 0.0, 0.0), reading(3.0, -6.0, 0.0)]))

    assert scores["over"].tolist() == [False, True]
    assert scores["score"].tolist()[1] > 1.0
    assert math.isinf(model.t2_limit)


def test_alarm_needs_three_of_the_last_five():
    over = pd.Series([False, True, True, False, True, False, False, False])

    assert persistent_alarm(over).tolist() == [False, False, False, False, True, True, False, False]


def test_model_survives_saving_and_loading(model):
    restored = pickle.loads(pickle.dumps(model.with_limits(10.0, 5.0)))

    assert restored.scores(reading(3.0, -6.0, 0.0))["over"].tolist() == [True]


def test_engine_split_is_fixed_disjoint_and_complete():
    units = np.arange(1, 101)

    fit, val, calibration = engine_split(units)

    assert (len(fit), len(val), len(calibration)) == (70, 15, 15)
    assert set(fit) | set(val) | set(calibration) == set(units)
    assert not set(fit) & set(val) and not set(val) & set(calibration)
    assert [list(part) for part in engine_split(units)] == [list(fit), list(val), list(calibration)]


def test_engine_split_matches_the_first_notebook():
    """Notebook 01 drew its engines with this exact recipe; the twin must keep the same ones."""
    shuffled = np.random.default_rng(42).permutation(np.arange(1, 101))

    fit, val, _ = engine_split(np.arange(1, 101))

    assert list(fit) == list(shuffled[:70])
    assert list(val) == list(shuffled[70:85])
