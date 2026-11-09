"""SQLite storage: one row per engine, life and cycle, with its reading, prediction and health."""

import json
import sqlite3
from pathlib import Path
from typing import Any, NamedTuple

SCHEMA_VERSION = 2
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
    t2 REAL,
    spe REAL,
    score REAL,
    over INTEGER,
    alarm INTEGER NOT NULL,
    top_sensors TEXT,
    PRIMARY KEY (engine_id, life, cycle)
)
"""
COLUMNS = (
    "engine_id", "life", "cycle", "settings", "sensors", "rul", "model", "status", "updated_at",
    "t2", "spe", "score", "over", "alarm", "top_sensors",
)  # fmt: skip
JSON_COLUMNS = {"settings", "sensors", "top_sensors"}


class StaleDatabase(Exception):
    """The database file was written by an older version of the twin."""


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
    t2: float | None = None
    spe: float | None = None
    score: float | None = None
    over: bool | None = None
    alarm: bool = False
    top_sensors: list[str] | None = None


def _to_row(raw: tuple) -> CycleRow:
    values: dict[str, Any] = dict(zip(COLUMNS, raw, strict=True))
    for name in JSON_COLUMNS:
        values[name] = None if values[name] is None else json.loads(values[name])
    values["alarm"] = bool(values["alarm"])
    values["over"] = None if values["over"] is None else bool(values["over"])
    return CycleRow(**values)


def _to_values(row: CycleRow) -> tuple:
    return tuple(
        json.dumps(value) if name in JSON_COLUMNS and value is not None else value
        for name, value in row._asdict().items()
    )


class TwinStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # The API calls this from worker threads; TwinRegistry holds a lock around every use.
        self._connection = sqlite3.connect(path, check_same_thread=False)
        version = self._connection.execute("PRAGMA user_version").fetchone()[0]
        has_table = self._connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'cycles'"
        ).fetchone()
        if has_table and version != SCHEMA_VERSION:
            self._connection.close()
            raise StaleDatabase(
                f"{path} was written by an older version of the twin. "
                "Start an empty twin with `pmt serve --fresh` or `pmt demo`."
            )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(SCHEMA)
        self._connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        self._connection.close()

    def save(self, rows: list[CycleRow]) -> None:
        placeholders = ", ".join("?" for _ in COLUMNS)
        with self._connection:
            self._connection.executemany(
                f"INSERT INTO cycles ({', '.join(COLUMNS)}) VALUES ({placeholders})",
                [_to_values(row) for row in rows],
            )

    def recent(self, engine_id: int, limit: int, life: int | None = None) -> list[CycleRow]:
        """Rows of one life (default: the latest), newest `limit`, oldest first."""
        if life is None:
            life = self._connection.execute(
                "SELECT MAX(life) FROM cycles WHERE engine_id = ?", (engine_id,)
            ).fetchone()[0]
        raw = self._connection.execute(
            f"SELECT {', '.join(COLUMNS)} FROM cycles WHERE engine_id = ? AND life = ? "
            "ORDER BY cycle DESC LIMIT ?",
            (engine_id, life, limit),
        ).fetchall()
        return [_to_row(item) for item in reversed(raw)]

    def engine_ids(self) -> list[int]:
        raw = self._connection.execute("SELECT DISTINCT engine_id FROM cycles ORDER BY 1")
        return [item[0] for item in raw]
