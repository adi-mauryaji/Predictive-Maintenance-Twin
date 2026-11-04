"""Command line entry point: `pmt <command>`."""

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from http.client import HTTPException
from urllib.parse import urlsplit

import httpx
import uvicorn

from pmtwin.config import PROJECT_ROOT, Settings
from pmtwin.data import CMAPSS_FILES, ChecksumError, download_cmapss, load_cmapss
from pmtwin.replay import replay

NOTEBOOK = "notebooks/01_first_model.ipynb"
DASHBOARD = PROJECT_ROOT / "dashboard" / "streamlit_app.py"


def _download(settings: Settings) -> None:
    try:
        download_cmapss(settings.raw_dir)
    except (ChecksumError, OSError, HTTPException) as error:
        raise SystemExit(f"Download failed: {error}") from error
    print(f"C-MAPSS is ready in {settings.raw_dir}")


def _forget_twin_state(settings: Settings) -> None:
    """Delete the twin's generated database so a replay can start again from cycle 1."""
    for suffix in ("", "-wal", "-shm"):
        settings.db_path.with_name(settings.db_path.name + suffix).unlink(missing_ok=True)


def _serve(settings: Settings, fresh: bool) -> None:
    if fresh:
        _forget_twin_state(settings)
    address = urlsplit(settings.api_url)
    uvicorn.run("pmtwin.api:app", host=address.hostname or "127.0.0.1", port=address.port or 8000)


def _replay(settings: Settings, subset: str, engines: int, tick: float) -> None:
    test = load_cmapss(settings.raw_dir, subset).test
    fleet = test.loc[test["unit"] <= engines].copy()
    with httpx.Client(base_url=settings.api_url, timeout=30) as client:
        try:
            sent = replay(client, fleet, tick)
        except httpx.HTTPStatusError as error:
            raise SystemExit(
                f"The twin refused a reading ({error.response.status_code}). "
                "If it already holds these cycles, restart it with `pmt serve --fresh`."
            ) from error
        except httpx.HTTPError as error:
            raise SystemExit(f"The twin API at {settings.api_url} is not reachable.") from error
    print(f"Replay finished: {sent} readings sent.")


def _wait_for_api(settings: Settings, api: subprocess.Popen) -> None:
    for _ in range(120):
        if api.poll() is not None:
            raise SystemExit(f"The API did not start. Is {settings.api_url} already in use?")
        try:
            httpx.get(f"{settings.api_url}/health", timeout=1).raise_for_status()
            return
        except httpx.HTTPError:
            time.sleep(0.25)
    raise SystemExit("The API did not answer in 30 seconds.")


def _demo(settings: Settings, subset: str, engines: int, tick: float) -> None:
    if not all((settings.raw_dir / name).exists() for name in CMAPSS_FILES):
        raise SystemExit("C-MAPSS data is missing. Run `pmt download` first.")
    if not settings.gbm_path.exists():
        raise SystemExit(f"The model is missing. Run {NOTEBOOK} to save it first.")
    _forget_twin_state(settings)
    address = urlsplit(settings.api_url)
    python = sys.executable
    api = subprocess.Popen(
        [python, "-m", "uvicorn", "pmtwin.api:app", "--log-level", "warning"]
        + ["--host", address.hostname or "127.0.0.1", "--port", str(address.port or 8000)]
    )
    dashboard = subprocess.Popen(
        [python, "-m", "streamlit", "run", str(DASHBOARD)]
        + ["--server.address", "127.0.0.1", "--browser.gatherUsageStats", "false"]
        + ["--client.toolbarMode", "minimal"],
        env={**os.environ, "PMT_API_URL": settings.api_url},
    )
    try:
        _wait_for_api(settings, api)
        _replay(settings, subset, engines, tick)
        print("The dashboard keeps running. Press Ctrl+C to stop everything.")
        dashboard.wait()
    except KeyboardInterrupt:
        pass
    finally:
        api.terminate()
        dashboard.terminate()


def _add_replay_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fd", default="FD001", help="C-MAPSS subset to replay")
    parser.add_argument("--engines", type=int, default=100, help="replay engines 1..N")
    parser.add_argument("--tick", type=float, default=0.2, help="seconds per cycle")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pmt", description="Predictive Maintenance Twin")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="fetch NASA C-MAPSS into data/raw and verify it")
    serve = commands.add_parser("serve", help="run the twin API")
    serve.add_argument("--fresh", action="store_true", help="start with an empty twin")
    _add_replay_options(commands.add_parser("replay", help="send recorded engines to the API"))
    _add_replay_options(
        commands.add_parser("demo", help="empty twin, then API, dashboard and replay together")
    )
    args = parser.parse_args(argv)

    settings = Settings()
    if args.command == "download":
        _download(settings)
    elif args.command == "serve":
        _serve(settings, args.fresh)
    elif args.command == "replay":
        _replay(settings, args.fd, args.engines, args.tick)
    elif args.command == "demo":
        _demo(settings, args.fd, args.engines, args.tick)
