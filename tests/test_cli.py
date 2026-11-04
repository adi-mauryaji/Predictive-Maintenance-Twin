"""`pmt download`: reports the configured folder, and failures end without a traceback."""

from http.client import IncompleteRead
from io import BytesIO
from urllib.error import URLError

import pytest

from pmtwin.cli import main
from pmtwin.data import CMAPSS_FILES


def test_download_reports_the_configured_folder(tmp_path, monkeypatch, capsys):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for name in CMAPSS_FILES:
        (raw_dir / name).write_text("already here")
    monkeypatch.setenv("PMT_DATA_DIR", str(tmp_path))

    main(["download"])

    assert str(raw_dir) in capsys.readouterr().out


def test_tampered_download_exits_without_traceback(tmp_path, monkeypatch):
    monkeypatch.setenv("PMT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("pmtwin.data.urlopen", lambda *args, **kwargs: BytesIO(b"tampered"))

    with pytest.raises(SystemExit) as exit_info:
        main(["download"])

    assert exit_info.value.code not in (0, None)


def test_network_failure_exits_without_traceback(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise URLError("no network")

    monkeypatch.setenv("PMT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("pmtwin.data.urlopen", no_network)

    with pytest.raises(SystemExit) as exit_info:
        main(["download"])

    assert exit_info.value.code not in (0, None)


def test_connection_dropped_mid_download_exits_without_traceback(tmp_path, monkeypatch):
    class DroppedConnection(BytesIO):
        def read(self, *args):
            raise IncompleteRead(b"partial")

    monkeypatch.setenv("PMT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("pmtwin.data.urlopen", lambda *args, **kwargs: DroppedConnection())

    with pytest.raises(SystemExit) as exit_info:
        main(["download"])

    assert exit_info.value.code not in (0, None)


def test_no_command_is_an_error():
    with pytest.raises(SystemExit) as exit_info:
        main([])

    assert exit_info.value.code not in (0, None)
