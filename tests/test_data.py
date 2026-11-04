"""Download rules: bad bytes never reach data/raw, and existing data needs no network."""

from hashlib import sha256
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from pmtwin.config import FD001_SENSORS, Settings
from pmtwin.data import (
    CMAPSS_FILES,
    ChecksumError,
    add_rul,
    download_cmapss,
    install_archive,
    load_cmapss,
)

DATA_FILES = [f"{kind}_FD00{i}.txt" for i in (1, 2, 3, 4) for kind in ("train", "test", "RUL")]


def make_archive():
    """Build a small archive shaped like the real one: a zip inside a folder inside a zip."""
    inner_buffer = BytesIO()
    with ZipFile(inner_buffer, "w") as inner:
        for name in DATA_FILES:
            inner.writestr(name, f"content of {name}")
        inner.writestr("Damage Propagation Modeling.pdf", "not data")
    inner_bytes = inner_buffer.getvalue()

    outer_buffer = BytesIO()
    with ZipFile(outer_buffer, "w") as outer:
        outer.writestr("6. Turbofan Data Set/CMAPSSData.zip", inner_bytes)
    outer_bytes = outer_buffer.getvalue()

    return outer_bytes, sha256(outer_bytes).hexdigest(), sha256(inner_bytes).hexdigest()


def test_install_writes_only_the_data_files(tmp_path):
    archive, outer_sha, inner_sha = make_archive()
    raw_dir = tmp_path / "raw"

    install_archive(archive, raw_dir, outer_sha, inner_sha)

    assert (raw_dir / "train_FD001.txt").read_bytes() == b"content of train_FD001.txt"
    assert (raw_dir / "RUL_FD004.txt").read_bytes() == b"content of RUL_FD004.txt"
    assert len(list(raw_dir.iterdir())) == 12


def test_wrong_outer_checksum_writes_nothing(tmp_path):
    archive, _, inner_sha = make_archive()
    raw_dir = tmp_path / "raw"

    with pytest.raises(ChecksumError):
        install_archive(archive, raw_dir, "0" * 64, inner_sha)

    assert not raw_dir.exists()


def test_wrong_inner_checksum_writes_nothing(tmp_path):
    archive, outer_sha, _ = make_archive()
    raw_dir = tmp_path / "raw"

    with pytest.raises(ChecksumError):
        install_archive(archive, raw_dir, outer_sha, "0" * 64)

    assert not raw_dir.exists()


def test_download_uses_no_network_when_data_exists(tmp_path, monkeypatch):
    for name in DATA_FILES:
        (tmp_path / name).write_text("already here")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("network used although the data exists")

    monkeypatch.setattr("pmtwin.data.urlopen", fail_if_called)

    download_cmapss(tmp_path)


def test_download_rejects_bytes_that_are_not_the_real_archive(tmp_path, monkeypatch):
    monkeypatch.setattr("pmtwin.data.urlopen", lambda *args, **kwargs: BytesIO(b"tampered"))

    with pytest.raises(ChecksumError):
        download_cmapss(tmp_path / "raw")

    assert not (tmp_path / "raw").exists()


def _row(unit, cycle, last_sensor):
    """One line of a C-MAPSS text file, with the trailing spaces the real files have."""
    numbers = [unit, cycle, 0.5, 0.25, 100.0, *range(1, 21), last_sensor]
    return " ".join(str(number) for number in numbers) + "  \n"


def test_load_names_every_column_without_shifting_values(tmp_path):
    (tmp_path / "train_FD001.txt").write_text(_row(1, 1, 21.5) + _row(1, 2, 22.5))
    (tmp_path / "test_FD001.txt").write_text(_row(1, 1, 31.5))
    (tmp_path / "RUL_FD001.txt").write_text("112 \n98 \n")

    subset = load_cmapss(tmp_path, "FD001")

    assert list(subset.train.columns[:6]) == [
        "unit",
        "cycle",
        "setting1",
        "setting2",
        "setting3",
        "s1",
    ]
    assert subset.train.columns[-1] == "s21"
    assert subset.train.shape == (2, 26)
    assert subset.train["cycle"].tolist() == [1, 2]
    assert subset.train["s21"].tolist() == [21.5, 22.5]
    assert subset.test["s21"].tolist() == [31.5]


def test_load_keys_true_rul_by_engine_number(tmp_path):
    (tmp_path / "train_FD001.txt").write_text(_row(1, 1, 0.0))
    (tmp_path / "test_FD001.txt").write_text(_row(1, 1, 0.0))
    (tmp_path / "RUL_FD001.txt").write_text("112 \n98 \n")

    subset = load_cmapss(tmp_path, "FD001")

    assert subset.test_rul.to_dict() == {1: 112, 2: 98}


@pytest.fixture
def real_raw_dir():
    raw_dir = Settings().raw_dir
    if not all((raw_dir / name).exists() for name in CMAPSS_FILES):
        pytest.skip("real data missing: run `pmt download`")
    return raw_dir


@pytest.mark.parametrize(
    ("name", "train_rows", "train_engines", "test_rows", "test_engines"),
    [
        ("FD001", 20631, 100, 13096, 100),
        ("FD002", 53759, 260, 33991, 259),
        ("FD003", 24720, 100, 16596, 100),
        ("FD004", 61249, 249, 41214, 248),
    ],
)
def test_real_files_have_the_verified_counts(
    real_raw_dir, name, train_rows, train_engines, test_rows, test_engines
):
    subset = load_cmapss(real_raw_dir, name)

    assert len(subset.train) == train_rows
    assert subset.train["unit"].nunique() == train_engines
    assert len(subset.test) == test_rows
    assert subset.test["unit"].nunique() == test_engines
    assert len(subset.test_rul) == test_engines
    assert int(subset.train.isna().sum().sum()) == 0
    assert int(subset.test.isna().sum().sum()) == 0


def test_real_fd001_first_engine_matches_the_files(real_raw_dir):
    subset = load_cmapss(real_raw_dir, "FD001")

    assert len(subset.train[subset.train["unit"] == 1]) == 192
    assert len(subset.test[subset.test["unit"] == 1]) == 31
    assert subset.test_rul.to_dict()[1] == 112


def test_add_rul_counts_down_to_zero_per_engine():
    engines = pd.DataFrame({"unit": [1, 1, 1, 2, 2], "cycle": [1, 2, 3, 1, 2]})

    labelled = add_rul(engines, rul_cap=125)

    assert labelled["rul"].tolist() == [2, 1, 0, 1, 0]


def test_add_rul_caps_only_the_target():
    engines = pd.DataFrame({"unit": [1] * 5, "cycle": [1, 2, 3, 4, 5]})

    labelled = add_rul(engines, rul_cap=3)

    assert labelled["rul"].tolist() == [4, 3, 2, 1, 0]
    assert labelled["target"].tolist() == [3, 3, 2, 1, 0]


def test_add_rul_leaves_the_input_unchanged():
    engines = pd.DataFrame({"unit": [1, 1], "cycle": [1, 2]})

    add_rul(engines, rul_cap=125)

    assert list(engines.columns) == ["unit", "cycle"]


def test_real_fd001_labels(real_raw_dir):
    train = load_cmapss(real_raw_dir, "FD001").train

    labelled = add_rul(train, Settings().rul_cap)

    first_engine = labelled[labelled["unit"] == 1]
    assert first_engine["rul"].tolist()[:2] == [191, 190]
    assert first_engine["target"].tolist()[:2] == [125, 125]
    assert labelled[labelled["rul"] == 0]["unit"].tolist() == list(range(1, 101))
    assert max(labelled["target"].tolist()) == 125


def test_fd001_sensor_list_keeps_exactly_the_sensors_that_vary(real_raw_dir):
    train = load_cmapss(real_raw_dir, "FD001").train
    sensors = [f"s{number}" for number in range(1, 22)]

    kept = [name for name in sensors if float(np.std(train[name].to_numpy(), ddof=1)) >= 0.01]

    assert tuple(kept) == FD001_SENSORS
