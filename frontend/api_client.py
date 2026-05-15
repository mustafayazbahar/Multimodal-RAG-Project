"""Thin HTTP client around the FastAPI backend."""
from __future__ import annotations

import json
import os
from typing import Iterator

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")


class ApiError(RuntimeError):
    pass


def _headers(token: str | None = None) -> dict:
    h = {"Accept": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def login(username: str, password: str) -> dict:
    resp = requests.post(
        f"{BACKEND_URL}/auth/login",
        json={"username": username, "password": password},
        timeout=10,
    )
    if resp.status_code != 200:
        raise ApiError(resp.json().get("detail", "Login failed"))
    return resp.json()


def register(username: str, password: str) -> dict:
    resp = requests.post(
        f"{BACKEND_URL}/auth/register",
        json={"username": username, "password": password},
        timeout=10,
    )
    if resp.status_code not in (200, 201):
        raise ApiError(resp.json().get("detail", "Register failed"))
    return resp.json()


def get_history(token: str) -> list[dict]:
    resp = requests.get(f"{BACKEND_URL}/chat/history", headers=_headers(token), timeout=10)
    if resp.status_code != 200:
        raise ApiError(resp.json().get("detail", "History failed"))
    return resp.json().get("messages", [])


def clear_history(token: str) -> None:
    resp = requests.delete(f"{BACKEND_URL}/chat/history", headers=_headers(token), timeout=10)
    if resp.status_code not in (200, 204):
        raise ApiError("Failed to clear history")


def list_models(token: str) -> dict:
    resp = requests.get(f"{BACKEND_URL}/chat/models", headers=_headers(token), timeout=10)
    if resp.status_code != 200:
        raise ApiError("Models lookup failed")
    return resp.json()


def pull_model(token: str, model: str) -> Iterator[dict]:
    """Stream pull-progress events from the backend."""
    with requests.post(
        f"{BACKEND_URL}/chat/models/pull",
        json={"model": model},
        headers=_headers(token),
        stream=True,
        timeout=None,
    ) as resp:
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", "")
            except json.JSONDecodeError:
                detail = resp.text
            raise ApiError(f"Pull failed: {detail}")
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def stream_query(
    token: str,
    query: str,
    model: str | None,
    temperature: float,
    top_k: int,
) -> Iterator[dict]:
    payload = {"query": query, "model": model, "temperature": temperature, "top_k": top_k}
    with requests.post(
        f"{BACKEND_URL}/chat/query",
        json=payload,
        headers=_headers(token),
        stream=True,
        timeout=600,
    ) as resp:
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", "")
            except json.JSONDecodeError:
                detail = resp.text
            raise ApiError(f"Query failed: {detail}")
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def ingest_status(token: str) -> dict:
    resp = requests.get(f"{BACKEND_URL}/ingest/status", headers=_headers(token), timeout=15)
    if resp.status_code != 200:
        raise ApiError("Status failed")
    return resp.json()


def upload_pdf(token: str, filename: str, data: bytes) -> dict:
    files = {"file": (filename, data, "application/pdf")}
    resp = requests.post(
        f"{BACKEND_URL}/ingest/upload",
        files=files,
        headers={"Authorization": f"Bearer {token}"},
        timeout=120,
    )
    if resp.status_code not in (200, 201):
        raise ApiError(resp.json().get("detail", "Upload failed"))
    return resp.json()


def run_ingest(token: str) -> dict:
    resp = requests.post(
        f"{BACKEND_URL}/ingest/run", headers=_headers(token), timeout=3600
    )
    if resp.status_code != 200:
        raise ApiError(resp.json().get("detail", "Ingest failed"))
    return resp.json()


def fetch_image_url(image_path: str) -> str:
    """Return the backend URL to fetch a given image path."""
    return f"{BACKEND_URL}/ingest/image?path={requests.utils.quote(image_path)}"
