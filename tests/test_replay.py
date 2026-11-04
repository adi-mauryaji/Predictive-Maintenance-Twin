"""The simulator sends one cycle per engine per tick, in order."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pmtwin.api import app, get_registry
from pmtwin.data import COLUMNS
from pmtwin.replay import replay


def small_fleet() -> pd.DataFrame:
    rows = [(1, 1), (1, 2), (1, 3), (2, 1)]
    records = [{name: 1.0 for name in COLUMNS} | {"unit": u, "cycle": c} for u, c in rows]
    return pd.DataFrame(records)


@pytest.fixture
def client(registry):
    app.dependency_overrides[get_registry] = lambda: registry
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_replay_brings_every_engine_to_its_last_cycle(client):
    ticks = []

    sent = replay(client, small_fleet(), tick_seconds=0.5, sleep=ticks.append)

    assert sent == 4
    assert ticks == [0.5, 0.5, 0.5]
    assert [(item["engine_id"], item["cycle"]) for item in client.get("/fleet").json()] == [
        (1, 3),
        (2, 1),
    ]


def test_replaying_twice_raises_the_conflict(client):
    replay(client, small_fleet(), tick_seconds=0, sleep=lambda _: None)

    with pytest.raises(Exception, match="409"):
        replay(client, small_fleet(), tick_seconds=0, sleep=lambda _: None)
