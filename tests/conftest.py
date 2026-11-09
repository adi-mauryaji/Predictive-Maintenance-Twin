"""Shared test helpers: a valid reading and a predictor whose answer the test controls."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pmtwin.config import Settings
from pmtwin.health import HealthModel
from pmtwin.store import TwinStore
from pmtwin.twin import TwinRegistry


class FakePredictor:
    """Predicts 100 minus the last s2 value, so a test picks the RUL through s2."""

    name = "fake"
    window = 3

    def __init__(self):
        self.windows = []

    def predict(self, windows):
        self.windows.extend(windows)
        return [100.0 - window[-1]["s2"] for window in windows]


def make_reading(engine_id: int = 1, cycle: int = 1, s2: float = 10.0, s3: float = 3.0) -> dict:
    sensors = {f"s{number}": float(number) for number in range(1, 22)}
    sensors["s2"] = s2
    sensors["s3"] = s3
    return {
        "engine_id": engine_id,
        "cycle": cycle,
        "settings": {"setting1": 0.0, "setting2": 0.0, "setting3": 100.0},
        "sensors": sensors,
    }


@pytest.fixture
def reading() -> Callable[..., dict]:
    return make_reading


@pytest.fixture
def predictor() -> FakePredictor:
    return FakePredictor()


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "twin.db"


@pytest.fixture
def registry(db_path, predictor):
    store = TwinStore(db_path)
    yield TwinRegistry(store, predictor, Settings())
    store.close()


@pytest.fixture(scope="session")
def health_model() -> HealthModel:
    """Normal s3 is about 3 and s4 about 4; s3 = 50 is far outside normal."""
    rng = np.random.default_rng(1)
    normal = pd.DataFrame({"s3": 3 + rng.normal(size=400), "s4": 4 + rng.normal(size=400)})
    scaler = StandardScaler().fit(normal)
    pca = PCA(n_components=0.95).fit(scaler.transform(normal))
    return HealthModel(["s3", "s4"], scaler, pca).with_limits(t2_limit=12.0, spe_limit=12.0)


@pytest.fixture
def health_registry(db_path, predictor, health_model):
    store = TwinStore(db_path)
    yield TwinRegistry(store, predictor, Settings(), health=health_model)
    store.close()
