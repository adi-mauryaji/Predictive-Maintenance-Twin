"""Sensor simulator: replays recorded engines cycle by cycle against the twin API."""

import time
from collections.abc import Callable

import httpx
import pandas as pd

SETTINGS = ("setting1", "setting2", "setting3")
SENSORS = tuple(f"s{number}" for number in range(1, 22))


def _reading(row: dict) -> dict:
    return {
        "engine_id": int(row["unit"]),
        "cycle": int(row["cycle"]),
        "settings": {name: row[name] for name in SETTINGS},
        "sensors": {name: row[name] for name in SENSORS},
    }


def replay(
    client: httpx.Client,
    engines: pd.DataFrame,
    tick_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Send each cycle of every engine, one tick per cycle. Returns the readings sent."""
    sent = 0
    for _, rows in engines.groupby("cycle", sort=True):
        readings = [_reading(row) for row in rows.to_dict("records")]
        client.post("/ingest", json={"readings": readings}).raise_for_status()
        sent += len(readings)
        sleep(tick_seconds)
    return sent
