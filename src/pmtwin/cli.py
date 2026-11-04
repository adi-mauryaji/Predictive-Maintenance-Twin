"""Command line entry point: `pmt <command>`."""

import argparse
from collections.abc import Sequence
from http.client import HTTPException

from pmtwin.config import Settings
from pmtwin.data import ChecksumError, download_cmapss


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pmt", description="Predictive Maintenance Twin")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="fetch NASA C-MAPSS into data/raw and verify it")
    args = parser.parse_args(argv)

    if args.command == "download":
        raw_dir = Settings().raw_dir
        try:
            download_cmapss(raw_dir)
        except (ChecksumError, OSError, HTTPException) as error:
            raise SystemExit(f"Download failed: {error}") from error
        print(f"C-MAPSS is ready in {raw_dir}")
