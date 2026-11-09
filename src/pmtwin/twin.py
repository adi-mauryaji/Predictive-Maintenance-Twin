"""The digital twin: live state per engine, kept in step with the database."""

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock

import pandas as pd

from pmtwin.config import Settings
from pmtwin.health import HealthModel, persistent_alarm
from pmtwin.predictors import RulPredictor
from pmtwin.schemas import (
    FleetItem,
    HealthResult,
    HistoryPoint,
    RulEstimate,
    SensorReading,
    Status,
    TwinSnapshot,
)
from pmtwin.store import CycleRow, TwinStore

ALARM_WINDOW = 5  # persistent_alarm looks at the last 5 readings
RECOMMENDATIONS = {
    Status.HEALTHY: "No action needed.",
    Status.WATCH: "Plan maintenance at the next opportunity.",
    Status.CRITICAL: "Schedule maintenance now.",
}
ALARM_RECOMMENDATION = (
    "Readings left their normal range in 3 of the last 5 cycles. Inspect at the next opportunity."
)


class CycleConflict(Exception):
    """A reading's cycle is not newer than what the twin already holds."""


class ModelUnavailable(Exception):
    """No trained model is loaded, so readings cannot be turned into predictions."""


def status_for(rul: float, alarm: bool, watch_rul: int, critical_rul: int) -> Status:
    if rul <= critical_rul:
        return Status.CRITICAL
    if rul <= watch_rul or alarm:
        return Status.WATCH
    return Status.HEALTHY


@dataclass
class _EngineState:
    life: int
    last_cycle: int
    buffer: deque[dict[str, float]]
    flags: deque[bool]
    snapshot: TwinSnapshot


class TwinRegistry:
    def __init__(
        self,
        store: TwinStore,
        predictor: RulPredictor | None,
        settings: Settings,
        health: HealthModel | None = None,
    ) -> None:
        self._store = store
        self.predictor = predictor
        self.health = health
        self._settings = settings
        self._window = predictor.window if predictor else 1
        self._lock = Lock()
        self._engines = {engine_id: self._restore(engine_id) for engine_id in store.engine_ids()}

    def _restore(self, engine_id: int) -> _EngineState:
        rows = self._store.recent(engine_id, limit=max(self._window, ALARM_WINDOW))
        latest = rows[-1]
        buffer = deque((row.sensors for row in rows[-self._window :]), maxlen=self._window)
        flags = deque((bool(row.over) for row in rows[-ALARM_WINDOW:]), maxlen=ALARM_WINDOW)
        return _EngineState(latest.life, latest.cycle, buffer, flags, self._snapshot_of(latest))

    def _snapshot_of(self, row: CycleRow) -> TwinSnapshot:
        status = Status(row.status)
        health = None
        if self.health is not None and row.score is not None:
            health = HealthResult(
                t2=row.t2 or 0.0,
                spe=row.spe or 0.0,
                score=row.score,
                t2_limit=self.health.t2_limit,
                spe_limit=self.health.spe_limit,
                alarm=row.alarm,
                top_sensors=row.top_sensors or [],
            )
        quiet_alarm = row.alarm and status == Status.WATCH and row.rul > self._settings.watch_rul
        return TwinSnapshot(
            engine_id=row.engine_id,
            life=row.life,
            cycle=row.cycle,
            rul=RulEstimate(value=row.rul, model=row.model),
            health=health,
            status=status,
            recommendation=ALARM_RECOMMENDATION if quiet_alarm else RECOMMENDATIONS[status],
            updated_at=datetime.fromisoformat(row.updated_at),
        )

    def ingest(self, readings: list[SensorReading]) -> list[TwinSnapshot]:
        """Apply a batch in order. Any bad reading rejects the whole batch and changes nothing."""
        if self.predictor is None:
            raise ModelUnavailable
        with self._lock:
            last_cycle = {key: state.last_cycle for key, state in self._engines.items()}
            buffers = {
                key: deque(state.buffer, self._window) for key, state in self._engines.items()
            }
            flags = {key: deque(state.flags, ALARM_WINDOW) for key, state in self._engines.items()}
            windows = []
            for reading in readings:
                if reading.cycle <= last_cycle.get(reading.engine_id, 0):
                    raise CycleConflict(
                        f"engine {reading.engine_id}: cycle {reading.cycle} is not newer than "
                        f"{last_cycle[reading.engine_id]}"
                    )
                last_cycle[reading.engine_id] = reading.cycle
                buffer = buffers.setdefault(reading.engine_id, deque(maxlen=self._window))
                buffer.append(reading.sensors)
                windows.append(list(buffer))

            predictions = self.predictor.predict(windows)
            health_scores: list[dict] = []
            tops: list[list[str]] = []
            if self.health is not None:
                frame = pd.DataFrame([reading.sensors for reading in readings])
                health_scores = self.health.scores(frame).to_dict("records")
                tops = self.health.top_sensors(frame)
            now = datetime.now(UTC).isoformat()
            rows = []
            for index, (reading, rul) in enumerate(zip(readings, predictions, strict=True)):
                extra = {}
                alarm = False
                if self.health is not None:
                    scores = health_scores[index]
                    engine_flags = flags.setdefault(reading.engine_id, deque(maxlen=ALARM_WINDOW))
                    engine_flags.append(bool(scores["over"]))
                    alarm = bool(persistent_alarm(pd.Series(list(engine_flags))).iloc[-1])
                    extra = {
                        "t2": float(scores["t2"]),
                        "spe": float(scores["spe"]),
                        "score": float(scores["score"]),
                        "over": bool(scores["over"]),
                        "top_sensors": tops[index],
                    }
                status = status_for(
                    rul, alarm, self._settings.watch_rul, self._settings.critical_rul
                )
                state = self._engines.get(reading.engine_id)
                rows.append(
                    CycleRow(
                        engine_id=reading.engine_id,
                        life=state.life if state else 1,
                        cycle=reading.cycle,
                        settings=reading.settings,
                        sensors=reading.sensors,
                        rul=rul,
                        model=self.predictor.name,
                        status=status.value,
                        updated_at=now,
                        alarm=alarm,
                        **extra,
                    )
                )
            self._store.save(rows)

            touched = {}
            for row in rows:
                snapshot = self._snapshot_of(row)
                self._engines[row.engine_id] = _EngineState(
                    row.life,
                    row.cycle,
                    buffers[row.engine_id],
                    flags.get(row.engine_id, deque(maxlen=ALARM_WINDOW)),
                    snapshot,
                )
                touched[row.engine_id] = snapshot
            return list(touched.values())

    def fleet(self) -> list[FleetItem]:
        with self._lock:
            return [
                FleetItem(
                    engine_id=engine_id,
                    life=state.life,
                    cycle=state.last_cycle,
                    status=state.snapshot.status,
                    rul=state.snapshot.rul.value,
                    alarm=bool(state.snapshot.health and state.snapshot.health.alarm),
                )
                for engine_id, state in sorted(self._engines.items())
            ]

    def snapshot(self, engine_id: int) -> TwinSnapshot | None:
        with self._lock:
            state = self._engines.get(engine_id)
            return state.snapshot if state else None

    def history(self, engine_id: int, limit: int) -> list[HistoryPoint]:
        with self._lock:
            rows = self._store.recent(engine_id, limit=limit)
        return [
            HistoryPoint(
                cycle=row.cycle,
                rul=row.rul,
                health_score=row.score,
                alarm=row.alarm,
                status=Status(row.status),
                sensors=row.sensors,
            )
            for row in rows
        ]
