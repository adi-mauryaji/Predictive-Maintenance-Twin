"""Project settings. A default can be overridden by an environment variable named PMT_<FIELD>."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# src/pmtwin/config.py -> src/pmtwin -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# The 14 FD001 sensors that vary; the rest are constant or noise (KYROS ROADMAP, ML protocol).
FD001_SENSORS = (
    "s2",
    "s3",
    "s4",
    "s7",
    "s8",
    "s9",
    "s11",
    "s12",
    "s13",
    "s14",
    "s15",
    "s17",
    "s20",
    "s21",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PMT_")

    data_dir: Path = PROJECT_ROOT / "data"
    rul_cap: int = 125

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"
