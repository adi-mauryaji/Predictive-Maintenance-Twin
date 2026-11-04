"""SQLite storage: one row per engine, life and cycle, with its reading and prediction."""

import json
import sqlite3
from pathlib import Path
from typing import NamedTuple

SCHEMA = """
CREATE TABLE IF NOT EXISTS cycles (
    engine_id INTEGER NOT NULL,
    life INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    settings TEXT NOT NULL,
    sensors TEXT NOT NULL,
    rul REAL NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (engine_id, life, cycle)
)
"""


class CycleRow(NamedTuple):
    engine_id: int
    life: int
    cycle: int
    settings: dict[str, float]
    sensors: dict[str, float]
    rul: float
    model: str
    status: str
    updated_at: str


def _to_row(raw: tuple) -> CycleRow:
    engine_id, life, cycle, settings, sensors, rul, model, status, updated_at = raw
    return CycleRow(
        engine_id,
        life,
        cycle,
        json.loads(settings),
        json.loads(sensors),
        rul,
        model,
        status,
        updated_at,
    )


class TwinStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # The API calls this from worker threads; TwinRegistry holds a lock around every use.
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(SCHEMA)

    def close(self) -> None:
        self._connection.close()

    def save(self, rows: list[CycleRow]) -> None:
        values = [
            (*row[:3], json.dumps(row.settings), json.dumps(row.sensors), *row[5:]) for row in rows
        ]
        with self._connection:
            self._connection.executemany(
                "INSERT INTO cycles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", values
            )

    def recent(self, engine_id: int, limit: int, life: int | None = None) -> list[CycleRow]:
        """Rows of one life (default: the latest), newest `limit`, oldest first."""
        if life is None:
            life = self._connection.execute(
                "SELECT MAX(life) FROM cycles WHERE engine_id = ?", (engine_id,)
            ).fetchone()[0]
        raw = self._connection.execute(
            "SELECT * FROM cycles WHERE engine_id = ? AND life = ? ORDER BY cycle DESC LIMIT ?",
            (engine_id, life, limit),
        ).fetchall()
        return [_to_row(item) for item in reversed(raw)]

    def engine_ids(self) -> list[int]:
        raw = self._connection.execute("SELECT DISTINCT engine_id FROM cycles ORDER BY 1")
        return [item[0] for item in raw]
