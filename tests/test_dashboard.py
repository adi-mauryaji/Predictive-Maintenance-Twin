"""The dashboard: it renders the API's data, and survives the API being down."""

from pathlib import Path

import httpx
import pandas as pd
import pytest
from charts import sensor_trend
from streamlit.testing.v1 import AppTest

ENTRY = str(Path(__file__).resolve().parents[1] / "dashboard" / "streamlit_app.py")

FLEET = [
    {"engine_id": 1, "life": 1, "cycle": 40, "status": "HEALTHY", "rul": 90.0},
    {"engine_id": 2, "life": 1, "cycle": 40, "status": "WATCH", "rul": 50.0},
    {"engine_id": 3, "life": 1, "cycle": 40, "status": "CRITICAL", "rul": 12.0},
]
SNAPSHOT = {
    "engine_id": 1,
    "life": 1,
    "cycle": 40,
    "rul": {"value": 90.0, "model": "lgbm"},
    "status": "HEALTHY",
    "recommendation": "No action needed.",
    "updated_at": "2026-11-04T10:00:00Z",
}
HISTORY = [
    {"cycle": 39, "rul": 91.0, "status": "HEALTHY", "sensors": {"s11": 47.1, "s12": 522.0}},
    {"cycle": 40, "rul": 90.0, "status": "HEALTHY", "sensors": {"s11": 47.2, "s12": 521.9}},
]


def fake_api(url, **kwargs):
    if url.endswith("/fleet"):
        body = FLEET
    elif url.endswith("/history"):
        body = HISTORY
    else:
        body = SNAPSHOT
    return httpx.Response(200, json=body, request=httpx.Request("GET", url))


def api_down(url, **kwargs):
    raise httpx.ConnectError("refused")


def run_app():
    return AppTest.from_file(ENTRY, default_timeout=30).run()


def test_fleet_page_counts_each_status(monkeypatch):
    monkeypatch.setattr(httpx, "get", fake_api)

    app = run_app()

    assert not app.exception
    assert [metric.value for metric in app.metric] == ["1", "1", "1"]
    assert app.dataframe[0].value["engine_id"].tolist() == [3, 2, 1]


def test_engine_page_opens_on_the_engine_closest_to_failure(monkeypatch):
    monkeypatch.setattr(httpx, "get", fake_api)

    app = run_app()
    app.switch_page("app_pages/engine.py").run()

    assert not app.exception
    assert app.selectbox[0].options == ["1", "2", "3"]
    assert app.selectbox[0].value == 3
    assert app.metric[0].value == "90"


def test_sensor_chart_axis_follows_the_data_so_drift_is_visible():
    points = pd.DataFrame({"cycle": [1, 2], "s12": [522.0, 519.5]})

    spec = sensor_trend(points, "s12").to_dict()

    assert spec["encoding"]["y"]["scale"]["zero"] is False


@pytest.mark.parametrize("page", ["app_pages/fleet.py", "app_pages/engine.py"])
def test_every_page_says_so_when_the_api_is_down(monkeypatch, page):
    monkeypatch.setattr(httpx, "get", api_down)

    app = run_app()
    app.switch_page(page).run()

    assert not app.exception
    assert app.warning


def test_fleet_page_waits_politely_before_the_first_reading(monkeypatch):
    def empty(url, **kwargs):
        return httpx.Response(200, json=[], request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", empty)

    app = run_app()

    assert not app.exception
    assert app.info
