"""All dashboard data comes through the twin's HTTP API, never from the models or the database."""

import os
from typing import Any

import httpx

API_URL = os.environ.get("PMT_API_URL", "http://127.0.0.1:8000")


class ApiUnavailable(Exception):
    """The twin API cannot be reached or refused the request."""


def get_json(path: str, **params: Any) -> Any:
    try:
        response = httpx.get(f"{API_URL}{path}", params=params, timeout=5)
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise ApiUnavailable(str(error)) from error
    return response.json()
