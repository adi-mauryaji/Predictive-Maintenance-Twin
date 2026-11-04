"""Settings must not depend on where a command is run from, and must be overridable."""

from pathlib import Path

from pmtwin.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_data_dir_ignores_current_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("PMT_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    assert Settings().data_dir == REPO_ROOT / "data"


def test_env_var_moves_the_raw_data_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("PMT_DATA_DIR", str(tmp_path))

    assert Settings().raw_dir == tmp_path / "raw"
