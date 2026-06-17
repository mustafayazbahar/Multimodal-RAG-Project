"""Thin HTTP client around the FastAPI backend."""
from __future__ import annotations

import json
import os
from typing import Iterator

import requests

# Backend FastAPI adresi ortam degiskeninden okunur; Docker icinde varsayilan
# "http://backend:8000". Sondaki "/" temizlenir ki URL birlestirmede cift
# slash olusmasin.
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")


# Backend cagrilarinda olusan tum hatalar bu ortak istisna tipiyle firlatilir;
# boylece arayuz katmani tek bir except blogu ile yakalayabilir.
class ApiError(RuntimeError):
    pass


# Istek basliklarini olusturan yardimci. Token verilirse Bearer yetkilendirme
# basligini ekler; verilmezse sadece JSON kabul basligi doner.
def _headers(token: str | None = None) -> dict:
    h = {"Accept": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


# Kullanici adi/parola ile backend uzerinden giris yapar ve token bilgisini doner.
def login(username: str, password: str) -> dict:
    resp = requests.post(
        f"{BACKEND_URL}/auth/login",
        json={"username": username, "password": password},
        timeout=10,
    )
    # 200 disindaki her durum basarisiz giris sayilir; backend'in dondurdugu
    # "detail" mesaji varsa onu, yoksa genel bir mesaji istisnaya tasiriz.
    if resp.status_code != 200:
        raise ApiError(resp.json().get("detail", "Login failed"))
    return resp.json()


def register(
    username: str,
    password: str,
    email: str,
    first_name: str = "",
    last_name: str = "",
) -> dict:
    """Create a Keycloak user via the backend Admin-API proxy.

    Keycloak requires email; first_name / last_name are optional but
    the form collects them for nicer realm-side records.
    """
    resp = requests.post(
        f"{BACKEND_URL}/auth/register",
        json={
            "username": username,
            "password": password,
            "email": email,
            "first_name": first_name,
            "last_name": last_name,
        },
        timeout=15,
    )
    # Basarili kayit 200 veya 201 doner. Hata govdesi her zaman JSON olmayabilir
    # (orn. proxy/gateway hatasi), bu yuzden cozumlemeyi try ile koruyoruz.
    if resp.status_code not in (200, 201):
        try:
            detail = resp.json().get("detail", "Register failed")
        except json.JSONDecodeError:
            detail = "Register failed"
        raise ApiError(detail)
    return resp.json()


def get_login_url(redirect_uri: str) -> str:
    """Ask the backend for the Keycloak `/auth` URL to redirect to."""
    resp = requests.get(
        f"{BACKEND_URL}/auth/login-url",
        params={"redirect_uri": redirect_uri},
        timeout=10,
    )
    if resp.status_code != 200:
        raise ApiError(resp.json().get("detail", "login-url failed"))
    return resp.json()["url"]


def exchange_code(code: str, redirect_uri: str) -> dict:
    """Trade a Keycloak callback `code` for a TokenResponse."""
    resp = requests.post(
        f"{BACKEND_URL}/auth/exchange-code",
        json={"code": code, "redirect_uri": redirect_uri},
        timeout=15,
    )
    if resp.status_code != 200:
        raise ApiError(resp.json().get("detail", "Code exchange failed"))
    return resp.json()


# Keycloak'tan cikis (logout) URL'sini ister. id_token_hint verilirse Keycloak
# onay sayfasi gostermeden sessiz cikis yapar; bu yuzden parametre opsiyonel.
def get_logout_url(redirect_uri: str, id_token_hint: str | None = None) -> str:
    params = {"redirect_uri": redirect_uri}
    # id_token elimizde varsa sorgu parametresi olarak ekleriz (sessiz logout icin).
    if id_token_hint:
        params["id_token_hint"] = id_token_hint
    resp = requests.get(
        f"{BACKEND_URL}/auth/logout-url",
        params=params,
        timeout=10,
    )
    if resp.status_code != 200:
        raise ApiError(resp.json().get("detail", "logout-url failed"))
    return resp.json()["url"]


def get_history(token: str, session_id: str | None = None) -> tuple[list[dict], str]:
    """Return (messages, resolved_session_id). Backend falls back to General Chat
    if `session_id` is None or stale."""
    params = {"session_id": session_id} if session_id else {}
    resp = requests.get(
        f"{BACKEND_URL}/chat/history",
        headers=_headers(token),
        params=params,
        timeout=10,
    )
    if resp.status_code != 200:
        raise ApiError(resp.json().get("detail", "History failed"))
    body = resp.json()
    return body.get("messages", []), body.get("session_id", "")


# Belirtilen oturumun (verilmezse aktif/varsayilan oturumun) sohbet gecmisini siler.
def clear_history(token: str, session_id: str | None = None) -> None:
    params = {"session_id": session_id} if session_id else {}
    resp = requests.delete(
        f"{BACKEND_URL}/chat/history",
        headers=_headers(token),
        params=params,
        timeout=10,
    )
    if resp.status_code not in (200, 204):
        raise ApiError("Failed to clear history")


# ─────────────────────────────────────────────────────────────────────────
# Sessions ("Topics")
# ─────────────────────────────────────────────────────────────────────────
# Kullanicinin tum sohbet oturumlarini ("Topics") listeler.
def list_sessions(token: str) -> list[dict]:
    resp = requests.get(
        f"{BACKEND_URL}/chat/sessions", headers=_headers(token), timeout=10
    )
    if resp.status_code != 200:
        raise ApiError(resp.json().get("detail", "Sessions failed"))
    return resp.json().get("sessions", [])


# Verilen baslikla yeni bir sohbet oturumu olusturur ve olusan oturumu doner.
def create_session(token: str, title: str) -> dict:
    resp = requests.post(
        f"{BACKEND_URL}/chat/sessions",
        json={"title": title},
        headers=_headers(token),
        timeout=10,
    )
    if resp.status_code not in (200, 201):
        raise ApiError(resp.json().get("detail", "Create session failed"))
    return resp.json()


# Mevcut bir oturumun basligini gunceller (PATCH ile kismi guncelleme).
def rename_session(token: str, session_id: str, title: str) -> dict:
    resp = requests.patch(
        f"{BACKEND_URL}/chat/sessions/{session_id}",
        json={"title": title},
        headers=_headers(token),
        timeout=10,
    )
    if resp.status_code != 200:
        raise ApiError(resp.json().get("detail", "Rename failed"))
    return resp.json()


# Verilen oturumu kalici olarak siler.
def delete_session(token: str, session_id: str) -> None:
    resp = requests.delete(
        f"{BACKEND_URL}/chat/sessions/{session_id}",
        headers=_headers(token),
        timeout=10,
    )
    # 204 (No Content) da basarili sayilir; silme sonrasi govde donmeyebilir.
    if resp.status_code not in (200, 204):
        try:
            detail = resp.json().get("detail", "Delete failed")
        except json.JSONDecodeError:
            detail = "Delete failed"
        raise ApiError(detail)


# Backend uzerinde kullanilabilir LLM modellerinin listesini doner.
def list_models(token: str) -> dict:
    resp = requests.get(f"{BACKEND_URL}/chat/models", headers=_headers(token), timeout=10)
    if resp.status_code != 200:
        raise ApiError("Models lookup failed")
    return resp.json()


def pull_model(token: str, model: str) -> Iterator[dict]:
    """Stream pull-progress events from the backend."""
    # Model indirme uzun surebilecegi icin stream=True ve timeout=None kullanilir;
    # backend her ilerleme olayini ayri bir JSON satiri olarak gonderir.
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
        # Gelen NDJSON akisini satir satir oku; bos satirlari ve bozuk JSON
        # parcalarini atlayarak her gecerli olayi cagirana yield ederiz.
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# RAG sorgusunu backend'e gonderir ve cevabi parca parca (streaming) doner.
# Cevap NDJSON akisidir: token parcalari, kaynaklar ve durum olaylari icerir.
def stream_query(
    token: str,
    query: str,
    model: str | None,
    temperature: float,
    top_k: int,
    session_id: str | None = None,
) -> Iterator[dict]:
    # Sorgu parametreleri tek bir JSON govdesinde toplanir.
    payload = {
        "query": query,
        "model": model,
        "temperature": temperature,
        "top_k": top_k,
        "session_id": session_id,
    }
    # stream=True ile yanit beklemeden akis baslar; timeout=600 uzun LLM
    # uretimleri icin genis tutulmustur.
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
        # NDJSON akisini satir satir cozumleyip her olayi cagirana aktariyoruz.
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# Ingest (belge isleme) surecinin guncel durumunu doner.
def ingest_status(token: str) -> dict:
    resp = requests.get(f"{BACKEND_URL}/ingest/status", headers=_headers(token), timeout=15)
    if resp.status_code != 200:
        raise ApiError("Status failed")
    return resp.json()


# PDF dosyasini multipart/form-data olarak backend'e yukler.
# Not: Content-Type'i requests'in dosya sinirlayicisini (boundary) kendisinin
# ayarlamasi icin _headers kullanmadan sadece Authorization basligi gonderiyoruz.
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


# Yuklenen belgeler icin ingest (parcalama + gomme + indeksleme) surecini baslatir.
# Islem cok uzun surebildiginden timeout 1 saat (3600 sn) olarak ayarlanmistir.
def run_ingest(token: str) -> dict:
    resp = requests.post(
        f"{BACKEND_URL}/ingest/run", headers=_headers(token), timeout=3600
    )
    if resp.status_code != 200:
        raise ApiError(resp.json().get("detail", "Ingest failed"))
    return resp.json()


# Bilgi tabanini (vektor deposu + islenmis belgeler) sifirlar.
def reset_knowledge_base(token: str) -> dict:
    resp = requests.post(
        f"{BACKEND_URL}/ingest/reset", headers=_headers(token), timeout=60
    )
    if resp.status_code != 200:
        raise ApiError(resp.json().get("detail", "Reset failed"))
    return resp.json()


def get_image_summaries(token: str) -> list[dict]:
    """Return Moondream image captions persisted during ingestion."""
    resp = requests.get(
        f"{BACKEND_URL}/ingest/image-summaries",
        headers=_headers(token),
        timeout=15,
    )
    if resp.status_code != 200:
        raise ApiError("Image summaries fetch failed")
    return resp.json()


def fetch_image_bytes(token: str, image_path: str) -> bytes | None:
    """Download an image through the backend."""
    resp = requests.get(
        f"{BACKEND_URL}/ingest/image",
        params={"path": image_path},
        headers=_headers(token),
        timeout=30,
    )
    if resp.status_code != 200:
        return None
    return resp.content
