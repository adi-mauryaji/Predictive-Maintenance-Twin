"""Download rules: bad bytes never reach data/raw, and existing data needs no network."""

from hashlib import sha256
from io import BytesIO
from zipfile import ZipFile

import pytest

from pmtwin.data import ChecksumError, download_cmapss, install_archive

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
