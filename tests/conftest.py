"""Shared test helpers: a valid reading and a predictor whose answer the test controls."""

from collections.abc import Callable

import pytest

from pmtwin.config import Settings
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


def make_reading(engine_id: int = 1, cycle: int = 1, s2: float = 10.0) -> dict:
    sensors = {f"s{number}": float(number) for number in range(1, 22)}
    sensors["s2"] = s2
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
