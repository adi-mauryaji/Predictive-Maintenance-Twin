"""HTTP contract: what a client sends, and what it gets back when something is wrong."""

import json

import pytest
from fastapi.testclient import TestClient

from pmtwin.api import app, get_registry
from pmtwin.config import Settings
from pmtwin.store import TwinStore
from pmtwin.twin import TwinRegistry


@pytest.fixture
def client(registry):
    app.dependency_overrides[get_registry] = lambda: registry
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_ingest_then_read_back(client, reading):
    response = client.post("/ingest", json={"readings": [reading(engine_id=4, cycle=1, s2=10.0)]})

    assert response.status_code == 200
    assert response.json()["snapshots"][0]["rul"]["value"] == 90.0
    assert client.get("/engines/4").json()["cycle"] == 1
    assert client.get("/engines/4/history").json()[0]["rul"] == 90.0


def test_fleet_lists_engines_in_order(client, reading):
    client.post("/ingest", json={"readings": [reading(engine_id=2), reading(engine_id=1, s2=80.0)]})

    fleet = client.get("/fleet").json()

    assert [item["engine_id"] for item in fleet] == [1, 2]
    assert fleet[0]["status"] == "CRITICAL"
    assert fleet[1]["status"] == "HEALTHY"


def test_old_cycle_is_a_conflict(client, reading):
    client.post("/ingest", json={"readings": [reading(cycle=5)]})

    assert client.post("/ingest", json={"readings": [reading(cycle=4)]}).status_code == 409


def test_unknown_engine_is_not_found(client):
    assert client.get("/engines/42").status_code == 404
    assert client.get("/engines/42/history").status_code == 404


@pytest.mark.parametrize("change", ["missing_sensor", "extra_sensor", "zero_cycle", "empty_batch"])
def test_bad_input_is_rejected(client, reading, change):
    item = reading()
    if change == "missing_sensor":
        del item["sensors"]["s21"]
    if change == "extra_sensor":
        item["sensors"]["s22"] = 1.0
    if change == "zero_cycle":
        item["cycle"] = 0
    readings = [] if change == "empty_batch" else [item]

    assert client.post("/ingest", json={"readings": readings}).status_code == 422


def test_infinite_sensor_value_is_rejected(client, reading):
    item = reading()
    item["sensors"]["s2"] = float("inf")
    body = json.dumps({"readings": [item]})

    response = client.post("/ingest", content=body, headers={"content-type": "application/json"})

    assert response.status_code == 422


def test_without_a_model_ingest_is_unavailable_but_reads_work(db_path, reading):
    store = TwinStore(db_path)
    app.dependency_overrides[get_registry] = lambda: TwinRegistry(store, None, Settings())
    client = TestClient(app)

    assert client.post("/ingest", json={"readings": [reading()]}).status_code == 503
    assert client.get("/fleet").json() == []
    assert client.get("/health").json()["status"] == "no_model"

    app.dependency_overrides.clear()
    store.close()


def test_health_names_the_loaded_model(client):
    assert client.get("/health").json() == {"status": "ok", "models": {"rul": "fake"}}
