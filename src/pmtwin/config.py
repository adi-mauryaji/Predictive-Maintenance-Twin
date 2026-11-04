"""Project settings. A default can be overridden by an environment variable named PMT_<FIELD>."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# src/pmtwin/config.py -> src/pmtwin -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PMT_")

    data_dir: Path = PROJECT_ROOT / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"
