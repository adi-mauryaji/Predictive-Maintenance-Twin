"""NASA C-MAPSS data: download with pinned checksums. Loading arrives at M1."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

# PHM Society mirror; the data.nasa.gov listing says unavailable (KYROS ADR-002).
CMAPSS_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
OUTER_SHA256 = "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"
INNER_SHA256 = "74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f"
INNER_ZIP_NAME = "CMAPSSData.zip"
CMAPSS_FILES = tuple(
    f"{kind}_FD00{subset}.txt" for subset in (1, 2, 3, 4) for kind in ("train", "test", "RUL")
)


class ChecksumError(ValueError):
    """Downloaded bytes do not match their pinned sha256."""


def _check_sha256(content: bytes, expected: str, label: str) -> None:
    actual = sha256(content).hexdigest()
    if actual != expected:
        raise ChecksumError(f"{label}: expected sha256 {expected}, got {actual}")


def install_archive(
    archive: bytes,
    raw_dir: Path,
    outer_sha256: str = OUTER_SHA256,
    inner_sha256: str = INNER_SHA256,
) -> None:
    """Verify the outer and inner zip, then write the data files. A mismatch writes nothing."""
    _check_sha256(archive, outer_sha256, "outer zip")
    with ZipFile(BytesIO(archive)) as outer:
        inner_path = next(name for name in outer.namelist() if name.endswith(INNER_ZIP_NAME))
        inner_bytes = outer.read(inner_path)
    _check_sha256(inner_bytes, inner_sha256, INNER_ZIP_NAME)

    raw_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(BytesIO(inner_bytes)) as inner:
        for name in CMAPSS_FILES:
            (raw_dir / name).write_bytes(inner.read(name))


def download_cmapss(raw_dir: Path) -> None:
    """Fetch and install C-MAPSS, unless raw_dir already holds every data file."""
    if all((raw_dir / name).exists() for name in CMAPSS_FILES):
        return
    with urlopen(CMAPSS_URL, timeout=120) as response:
        install_archive(response.read(), raw_dir)
