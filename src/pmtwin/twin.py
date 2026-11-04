"""The digital twin: live state per engine, kept in step with the database."""

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock

from pmtwin.config import Settings
from pmtwin.predictors import RulPredictor
from pmtwin.schemas import FleetItem, HistoryPoint, RulEstimate, SensorReading, Status, TwinSnapshot
from pmtwin.store import CycleRow, TwinStore

RECOMMENDATIONS = {
    Status.HEALTHY: "No action needed.",
    Status.WATCH: "Plan maintenance at the next opportunity.",
    Status.CRITICAL: "Schedule maintenance now.",
}


class CycleConflict(Exception):
    """A reading's cycle is not newer than what the twin already holds."""


class ModelUnavailable(Exception):
    """No trained model is loaded, so readings cannot be turned into predictions."""


def status_for(rul: float, watch_rul: int, critical_rul: int) -> Status:
    if rul <= critical_rul:
        return Status.CRITICAL
    if rul <= watch_rul:
        return Status.WATCH
    return Status.HEALTHY


@dataclass
class _EngineState:
    life: int
    last_cycle: int
    buffer: deque[dict[str, float]]
    snapshot: TwinSnapshot


def _snapshot_of(row: CycleRow) -> TwinSnapshot:
    status = Status(row.status)
    return TwinSnapshot(
        engine_id=row.engine_id,
        life=row.life,
        cycle=row.cycle,
        rul=RulEstimate(value=row.rul, model=row.model),
        status=status,
        recommendation=RECOMMENDATIONS[status],
        updated_at=datetime.fromisoformat(row.updated_at),
    )


class TwinRegistry:
    def __init__(
        self, store: TwinStore, predictor: RulPredictor | None, settings: Settings
    ) -> None:
        self._store = store
        self.predictor = predictor
        self._settings = settings
        self._window = predictor.window if predictor else 1
        self._lock = Lock()
        self._engines = {engine_id: self._restore(engine_id) for engine_id in store.engine_ids()}

    def _restore(self, engine_id: int) -> _EngineState:
        rows = self._store.recent(engine_id, limit=self._window)
        latest = rows[-1]
        buffer = deque((row.sensors for row in rows), maxlen=self._window)
        return _EngineState(latest.life, latest.cycle, buffer, _snapshot_of(latest))

    def ingest(self, readings: list[SensorReading]) -> list[TwinSnapshot]:
        """Apply a batch in order. Any bad reading rejects the whole batch and changes nothing."""
        if self.predictor is None:
            raise ModelUnavailable
        with self._lock:
            last_cycle = {key: state.last_cycle for key, state in self._engines.items()}
            buffers = {
                key: deque(state.buffer, self._window) for key, state in self._engines.items()
            }
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
            now = datetime.now(UTC).isoformat()
            rows = []
            for reading, rul in zip(readings, predictions, strict=True):
                status = status_for(rul, self._settings.watch_rul, self._settings.critical_rul)
                life = (
                    self._engines[reading.engine_id].life
                    if reading.engine_id in self._engines
                    else 1
                )
                rows.append(
                    CycleRow(
                        reading.engine_id,
                        life,
                        reading.cycle,
                        reading.settings,
                        reading.sensors,
                        rul,
                        self.predictor.name,
                        status.value,
                        now,
                    )
                )
            self._store.save(rows)

            touched = {}
            for row in rows:
                snapshot = _snapshot_of(row)
                self._engines[row.engine_id] = _EngineState(
                    row.life, row.cycle, buffers[row.engine_id], snapshot
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
                cycle=row.cycle, rul=row.rul, status=Status(row.status), sensors=row.sensors
            )
            for row in rows
        ]
