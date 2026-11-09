"""The twin: state per engine, order of cycles, and surviving a restart."""

import joblib
import numpy as np
import pytest

from pmtwin.config import FD001_SENSORS, Settings
from pmtwin.data import CMAPSS_FILES, load_cmapss
from pmtwin.features import add_rolling_means
from pmtwin.predictors import GbmPredictor
from pmtwin.schemas import SensorReading, Status
from pmtwin.store import StaleDatabase, TwinStore
from pmtwin.twin import CycleConflict, TwinRegistry, status_for


def readings_of(*raw):
    return [SensorReading(**item) for item in raw]


def test_ingest_returns_the_new_state_of_the_engine(registry, reading):
    snapshot = registry.ingest(readings_of(reading(engine_id=7, cycle=1, s2=10.0)))[0]

    assert (snapshot.engine_id, snapshot.life, snapshot.cycle) == (7, 1, 1)
    assert snapshot.rul.value == 90.0
    assert snapshot.rul.model == "fake"
    assert snapshot.status == Status.HEALTHY


@pytest.mark.parametrize(
    ("rul", "expected"),
    [
        (61.0, Status.HEALTHY),
        (60.0, Status.WATCH),
        (31.0, Status.WATCH),
        (30.0, Status.CRITICAL),
        (0.0, Status.CRITICAL),
    ],
)
def test_status_follows_the_rul_thresholds(rul, expected):
    assert status_for(rul, alarm=False, watch_rul=60, critical_rul=30) == expected


def test_alarm_alone_is_a_watch_and_low_rul_stays_critical():
    assert status_for(100.0, alarm=True, watch_rul=60, critical_rul=30) == Status.WATCH
    assert status_for(10.0, alarm=True, watch_rul=60, critical_rul=30) == Status.CRITICAL


def test_alarm_rings_on_the_third_of_five_bad_readings(health_registry, reading):
    snapshots = [
        health_registry.ingest(readings_of(reading(cycle=cycle, s3=s3)))[0]
        for cycle, s3 in enumerate([50.0, 3.0, 50.0, 50.0], start=1)
    ]
    snapshot = snapshots[-1]

    assert [item.health.alarm for item in snapshots if item.health] == [False, False, False, True]
    assert snapshot.health is not None
    assert snapshot.status == Status.WATCH
    assert snapshot.rul.value == 90.0
    assert snapshot.health.top_sensors[0] == "s3"
    assert "normal" in snapshot.recommendation


def test_alarm_history_survives_a_restart(db_path, predictor, health_model, reading):
    store = TwinStore(db_path)
    first = TwinRegistry(store, predictor, Settings(), health=health_model)
    for cycle in (1, 2):
        first.ingest(readings_of(reading(cycle=cycle, s3=50.0)))
    store.close()

    store = TwinStore(db_path)
    restarted = TwinRegistry(store, predictor, Settings(), health=health_model)
    snapshot = restarted.ingest(readings_of(reading(cycle=3, s3=50.0)))[0]
    store.close()

    assert snapshot.health is not None
    assert snapshot.health.alarm is True


def test_without_a_health_model_there_is_no_alarm(registry, reading):
    snapshot = registry.ingest(readings_of(reading(s3=50.0)))[0]

    assert snapshot.health is None
    assert snapshot.status == Status.HEALTHY


def test_an_old_database_is_refused_with_a_clear_message(db_path):
    import sqlite3

    old = sqlite3.connect(db_path)
    old.execute("CREATE TABLE cycles (engine_id INTEGER)")
    old.close()

    with pytest.raises(StaleDatabase, match="--fresh"):
        TwinStore(db_path)


def test_old_cycle_is_rejected_and_changes_nothing(registry, reading):
    registry.ingest(readings_of(reading(engine_id=1, cycle=5)))

    with pytest.raises(CycleConflict):
        registry.ingest(
            readings_of(reading(engine_id=2, cycle=1), reading(engine_id=1, cycle=5, s2=99.0))
        )

    assert [item.engine_id for item in registry.fleet()] == [1]
    snapshot = registry.snapshot(1)
    assert snapshot is not None
    assert snapshot.rul.value == 90.0
    assert len(registry.history(1, limit=10)) == 1


def test_predictor_sees_only_the_last_window_of_that_engine(registry, predictor, reading):
    for cycle in (1, 2, 3, 4):
        registry.ingest(readings_of(reading(engine_id=1, cycle=cycle, s2=float(cycle))))
    registry.ingest(readings_of(reading(engine_id=2, cycle=1, s2=50.0)))

    assert [row["s2"] for row in predictor.windows[3]] == [2.0, 3.0, 4.0]
    assert [row["s2"] for row in predictor.windows[4]] == [50.0]


def test_two_cycles_of_one_engine_in_one_batch(registry, predictor, reading):
    snapshots = registry.ingest(readings_of(reading(cycle=1, s2=1.0), reading(cycle=2, s2=2.0)))

    assert [row["s2"] for row in predictor.windows[1]] == [1.0, 2.0]
    assert [(item.cycle, item.rul.value) for item in snapshots] == [(2, 98.0)]


def test_restart_rebuilds_state_from_the_database(db_path, predictor, reading):
    store = TwinStore(db_path)
    first = TwinRegistry(store, predictor, Settings())
    for cycle in (1, 2, 3, 4):
        first.ingest(readings_of(reading(engine_id=3, cycle=cycle, s2=float(cycle))))
    store.close()

    store = TwinStore(db_path)
    restarted = TwinRegistry(store, predictor, Settings())

    snapshot = restarted.snapshot(3)
    assert snapshot is not None
    assert (snapshot.cycle, snapshot.rul.value) == (4, 96.0)
    with pytest.raises(CycleConflict):
        restarted.ingest(readings_of(reading(engine_id=3, cycle=4)))
    restarted.ingest(readings_of(reading(engine_id=3, cycle=5, s2=5.0)))
    assert [row["s2"] for row in predictor.windows[-1]] == [3.0, 4.0, 5.0]
    store.close()


def test_history_is_oldest_first_and_honours_the_limit(registry, reading):
    for cycle in (1, 2, 3):
        registry.ingest(readings_of(reading(cycle=cycle, s2=float(cycle))))

    points = registry.history(1, limit=2)

    assert [(point.cycle, point.rul) for point in points] == [(2, 98.0), (3, 97.0)]
    assert points[0].sensors["s2"] == 2.0


def test_negative_predictions_become_zero():
    class AlwaysNegative:
        def predict(self, frame):
            return np.array([-3.0] * len(frame))

    predictor = GbmPredictor(AlwaysNegative(), ["s2", "s2_mean"], window=2)

    assert predictor.predict([[{"s2": 1.0}]]) == [0.0]


def test_twin_health_score_equals_the_notebook_score(db_path):
    settings = Settings()
    has_data = all((settings.raw_dir / name).exists() for name in CMAPSS_FILES)
    if not (has_data and settings.gbm_path.exists() and settings.health_path.exists()):
        pytest.skip("needs `pmt download` and both notebooks' artifacts")
    health = joblib.load(settings.health_path)
    engine = load_cmapss(settings.raw_dir, "FD001").test.query("unit == 1").reset_index(drop=True)
    sensors = [f"s{number}" for number in range(1, 22)]
    store = TwinStore(db_path)
    registry = TwinRegistry(store, GbmPredictor.load(settings.gbm_path), settings, health=health)

    served = []
    for row in engine.to_dict("records"):
        reading = SensorReading(
            engine_id=1,
            cycle=int(row["cycle"]),
            settings={name: row[name] for name in ("setting1", "setting2", "setting3")},
            sensors={name: row[name] for name in sensors},
        )
        result = registry.ingest([reading])[0].health
        assert result is not None
        served.append(result.score)
    store.close()

    assert np.allclose(served, health.scores(engine)["score"])


def test_twin_gives_the_same_answer_as_training_features(db_path):
    """Serving must compute features exactly as training did."""
    settings = Settings()
    has_data = all((settings.raw_dir / name).exists() for name in CMAPSS_FILES)
    if not (has_data and settings.gbm_path.exists()):
        pytest.skip("needs `pmt download` and the notebook's artifact")
    predictor = GbmPredictor.load(settings.gbm_path)
    engine = load_cmapss(settings.raw_dir, "FD001").test.query("unit == 1").reset_index(drop=True)
    sensors = [f"s{number}" for number in range(1, 22)]
    store = TwinStore(db_path)
    registry = TwinRegistry(store, predictor, settings)

    served = []
    for row in engine.to_dict("records"):
        reading = SensorReading(
            engine_id=1,
            cycle=int(row["cycle"]),
            settings={name: row[name] for name in ("setting1", "setting2", "setting3")},
            sensors={name: row[name] for name in sensors},
        )
        served.append(registry.ingest([reading])[0].rul.value)
    store.close()

    trained = add_rolling_means(engine, list(FD001_SENSORS), predictor.window)
    direct = predictor.model.predict(trained[predictor.features])
    assert np.allclose(served, np.clip(direct, 0, None))
