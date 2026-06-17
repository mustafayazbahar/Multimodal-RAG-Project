"""Keycloak OIDC client + Admin API wrapper.

Replaces the legacy SQLite + bcrypt + locally-signed JWT auth. The
identity provider is Keycloak (a separate container, port 8080); our
backend just validates JWTs signed by the `deepcampus` realm and
proxies user creation to the Keycloak Admin API.

The chat_history table (services/auth.py) is still a local SQLite —
it's user-keyed data, not authentication, and we don't want to push it
into Keycloak.

Token flow:
1. Frontend posts {username, password} to backend /auth/login.
2. Backend calls Keycloak's password-grant token endpoint and gets a
   signed access_token back. We verify it locally against the realm's
   JWKS (no plaintext password ever leaves the backend), pull the role
   out of `realm_access.roles`, and ship the token to the frontend.
3. Frontend stashes the token in localStorage (see frontend/session.py)
   and sends it in `Authorization: Bearer ...` on every API call.
4. Backend (backend/security.py) re-verifies the token on each request
   via verify_token() below.

Role model: we look at `realm_access.roles`. If 'instructor' is present
the user is an instructor, otherwise student. Defaults to student.
"""
from __future__ import annotations

import json
import urllib.parse
from functools import lru_cache

import jwt
import requests
from jwt.algorithms import RSAAlgorithm

from services.config import settings
from services.logging_config import get_logger

log = get_logger(__name__)


class KeycloakError(RuntimeError):
    """Raised when Keycloak token or Admin API calls fail."""


# Asagidaki _*_url yardimcilari Keycloak OIDC uc noktalarini uretir.
# KRITIK: Sunucudan-sunucuya cagrilar ic adresi (settings.keycloak.url),
# tarayiciya yonlendirilen baglantilar ise public_url'i kullanir.
def _token_url() -> str:
    # Token uc noktasi (backend tarafi). Hem password grant hem code exchange burayi cagirir.
    return (
        f"{settings.keycloak.url}/realms/{settings.keycloak.realm}"
        f"/protocol/openid-connect/token"
    )


def _public_auth_url() -> str:
    """Browser-facing authorize endpoint. Uses the public Keycloak host."""
    # Tarayicinin gidecegi yetkilendirme adresi; public_url kullanir cunku
    # kullanici tarayicisi Docker ic ag adini cozemez.
    return (
        f"{settings.keycloak.public_url}/realms/{settings.keycloak.realm}"
        f"/protocol/openid-connect/auth"
    )


def _public_logout_url() -> str:
    """Browser-facing end-session endpoint."""
    # Cikis (end-session) adresi; yine tarayici tarafi oldugu icin public_url.
    return (
        f"{settings.keycloak.public_url}/realms/{settings.keycloak.realm}"
        f"/protocol/openid-connect/logout"
    )


def _certs_url() -> str:
    # JWKS (imza anahtarlari) adresi. Backend, JWT imzasini dogrularken kullanir; ic adres.
    return (
        f"{settings.keycloak.url}/realms/{settings.keycloak.realm}"
        f"/protocol/openid-connect/certs"
    )


def _admin_token_url() -> str:
    # Admin API icin token; master realm'den alinir (kullanici yonetimi yetkisi buradadir).
    return f"{settings.keycloak.url}/realms/master/protocol/openid-connect/token"


def _admin_realm_url() -> str:
    # Admin API'nin deepcampus realm'i icin temel adresi (kullanici/rol islemleri buraya gider).
    return f"{settings.keycloak.url}/admin/realms/{settings.keycloak.realm}"


# ─────────────────────────────────────────────────────────────────────────
# Login (password grant)
# ─────────────────────────────────────────────────────────────────────────
def login(username: str, password: str) -> dict:
    """Get a Keycloak access token via the password grant flow.

    Returns the full token response (access_token, refresh_token,
    expires_in, ...). Raises KeycloakError on bad credentials or
    network failure.
    """
    # Password grant: kullanici adi+parola dogrudan Keycloak'a gonderilir ve token alinir.
    data = {
        "grant_type": "password",
        "client_id": settings.keycloak.client_id,
        "username": username,
        "password": password,
    }
    # Ag hatalarini (Keycloak ayakta degil/erisilemez) anlamli bir KeycloakError'a ceviriyoruz.
    try:
        resp = requests.post(_token_url(), data=data, timeout=10)
    except requests.RequestException as exc:
        raise KeycloakError(f"Keycloak unreachable: {exc}") from exc

    # 200 disindaki yanitlar hatali kimlik bilgisi demektir. Keycloak'in
    # error_description alanini ayiklamaya calisir, yoksa genel bir mesaja duseriz.
    if resp.status_code != 200:
        try:
            detail = resp.json().get("error_description", "Invalid credentials")
        except (json.JSONDecodeError, ValueError):
            detail = "Invalid credentials"
        raise KeycloakError(detail)
    return resp.json()


# ─────────────────────────────────────────────────────────────────────────
# Authorization Code flow (browser-initiated login)
# ─────────────────────────────────────────────────────────────────────────
def build_login_url(redirect_uri: str) -> str:
    """Construct the Keycloak `/auth` URL the browser should hit.

    Uses the public Keycloak hostname (not the in-Docker hostname) so
    the user's browser can actually reach Keycloak. The realm config
    has standardFlowEnabled=true and redirectUris=['*'], so any
    callback URL the frontend chooses is accepted.
    """
    # OAuth Authorization Code akisinin ilk adimi: tarayicinin yonlendirilecegi URL.
    # response_type=code ile Keycloak, basarili girisin ardindan redirect_uri'ye bir "code" dondurur.
    params = {
        "client_id": settings.keycloak.client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": "openid profile email",
    }
    # public_auth_url + parametreler; urlencode ozel karakterleri guvenli sekilde kacirir.
    return f"{_public_auth_url()}?{urllib.parse.urlencode(params)}"


def exchange_code(code: str, redirect_uri: str) -> dict:
    """Trade a callback `code` for the realm token bundle.

    Backend-side call (server-to-server) so it uses the internal
    Keycloak hostname. Returns the full token response, including
    `id_token` which is later required for a silent logout.
    """
    # Code akisinin ikinci adimi: tarayicidan donen "code", backend tarafindan
    # (sunucudan-sunucuya, bu yuzden ic adres _token_url) token paketine takas edilir.
    # redirect_uri ilk adimdakiyle birebir ayni olmali, yoksa Keycloak reddeder.
    data = {
        "grant_type": "authorization_code",
        "client_id": settings.keycloak.client_id,
        "code": code,
        "redirect_uri": redirect_uri,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    try:
        resp = requests.post(_token_url(), data=data, headers=headers, timeout=10)
    except requests.RequestException as exc:
        raise KeycloakError(f"Keycloak unreachable: {exc}") from exc
    # Basarisiz takas (orn. suresi gecmis ya da tekrar kullanilan code) -> KeycloakError.
    if resp.status_code != 200:
        try:
            detail = resp.json().get("error_description", "Code exchange failed")
        except (json.JSONDecodeError, ValueError):
            detail = "Code exchange failed"
        raise KeycloakError(detail)
    return resp.json()


def build_logout_url(redirect_uri: str, id_token_hint: str | None = None) -> str:
    """Construct the Keycloak end-session URL for a one-tap logout.

    `id_token_hint` skips the Keycloak "confirm logout" page; without it
    Keycloak will ask the user to confirm. We pass `client_id` too as a
    fallback for the no-id-token case.
    """
    # Cikis URL'i. id_token_hint varsa Keycloak "cikisi onayla" ekranini atlar (tek tikla cikis);
    # yoksa client_id'yi yedek olarak gondeririz ki Keycloak istegi yine de iliskilendirebilsin.
    params: dict[str, str] = {
        "post_logout_redirect_uri": redirect_uri,
        "client_id": settings.keycloak.client_id,
    }
    if id_token_hint:
        params["id_token_hint"] = id_token_hint
    return f"{_public_logout_url()}?{urllib.parse.urlencode(params)}"


# ─────────────────────────────────────────────────────────────────────────
# JWT verification
# ─────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _jwks_cached() -> dict:
    """Fetch and cache the realm's JWKS payload.

    Keycloak rotates signing keys infrequently. We cache the entire
    JWKS in-process to avoid an HTTP round-trip on every request. If
    a key rotation happens the verify_token call will fall back to a
    cache invalidation via _refresh_jwks().
    """
    # JWKS'i (realm imza anahtarlari) cekip lru_cache ile bellekte tutar.
    # Anahtarlar nadiren degistigi icin her istekte HTTP cagrisi yapmaktan kacinir.
    resp = requests.get(_certs_url(), timeout=10)
    resp.raise_for_status()
    return resp.json()


def _refresh_jwks() -> dict:
    """Force-refresh the JWKS cache (e.g. after a kid miss)."""
    # Onbellegi temizleyip JWKS'i yeniden ceker. Keycloak anahtar dondurdugunde
    # (rotation) eski onbellekte olmayan yeni kid'i yakalamak icin kullanilir.
    _jwks_cached.cache_clear()
    return _jwks_cached()


def _public_key_for_kid(kid: str) -> object | None:
    """Return the loaded RSA public key for the given kid, or None."""
    # Token'in kid'ine (anahtar kimligi) uyan RSA public key'i bulur.
    # Onemli: once onbellekli JWKS'e bakariz; bulunamazsa _refresh_jwks ile
    # taze anahtarlari deneriz. Bu, anahtar rotasyonu sonrasi ilk dogrulamanin
    # bos onbellek yuzunden basarisiz olmasini onler.
    for jwks in (_jwks_cached(), _refresh_jwks()):
        for k in jwks.get("keys", []):
            if k.get("kid") == kid:
                # JWK formatindaki anahtari PyJWT'nin kullanabilecegi RSA nesnesine cevir.
                return RSAAlgorithm.from_jwk(json.dumps(k))
    return None


def verify_token(token: str) -> dict:
    """Verify a Keycloak-signed JWT and return its claims.

    Signature is checked against the realm's published JWKS. Audience
    is NOT pinned (Keycloak ships multiple aud entries — 'account',
    the client id, etc. — and our trust boundary is "any JWT issued by
    our realm" which the signature already guarantees).
    """
    # Once imzayi DOGRULAMADAN sadece basligi okuyup hangi anahtarin (kid)
    # kullanildigini ogreniriz. Bozuk/ayrismayan token burada yakalanir.
    try:
        header = jwt.get_unverified_header(token)
    except jwt.InvalidTokenError as exc:
        raise KeycloakError(f"Malformed token: {exc}") from exc

    # kid olmadan dogru anahtari secemeyiz; eksikse reddet.
    kid = header.get("kid")
    if not kid:
        raise KeycloakError("Token missing kid header")

    # Bu kid'e karsilik gelen realm public key'ini bul; yoksa token bizim realm'imizden degildir.
    public_key = _public_key_for_kid(kid)
    if public_key is None:
        raise KeycloakError(f"No realm signing key matches kid={kid}")

    # Asil dogrulama: RS256 imzasini public key ile kontrol et ve claim'leri dondur.
    # verify_aud=False bilincli bir tercih: Keycloak birden cok aud uretir ve guven
    # sinirimiz "realm tarafindan imzalanmis herhangi bir token" oldugu icin imza yeterli.
    try:
        return jwt.decode(
            token,
            key=public_key,
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
    # Hata turlerini ayri ayri anlamli mesajlara ceviriyoruz (suresi gecmis vs. genel gecersiz).
    except jwt.ExpiredSignatureError as exc:
        raise KeycloakError("Token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise KeycloakError(f"Invalid token: {exc}") from exc


def extract_user(claims: dict) -> dict:
    """Pull the {username, role} shape our backend uses out of a JWT.

    Role precedence: any user with 'instructor' in realm_access.roles
    is treated as an instructor; everyone else is a student. This
    matches the realm's defaultRoles wiring.
    """
    # Kullanici adi: tercihen preferred_username, yoksa sub (benzersiz kullanici kimligi).
    username = claims.get("preferred_username") or claims.get("sub", "")
    # Roller realm_access.roles altinda gelir; claim eksik olabilecegi icin (or {}) / (or [])
    # ile guvenli erisim yapariz.
    roles = (claims.get("realm_access") or {}).get("roles") or []
    # Rol onceligi: 'instructor' rolu varsa egitmen, aksi halde herkes ogrenci (varsayilan).
    role = "instructor" if "instructor" in roles else "student"
    return {"username": username, "role": role}


# ─────────────────────────────────────────────────────────────────────────
# Admin API (user registration)
# ─────────────────────────────────────────────────────────────────────────
def _get_admin_token() -> str:
    """Get a master-realm admin token for Admin API calls."""
    # Admin API'yi cagirabilmek icin master realm'den yetkili bir token alir.
    # admin-cli, Keycloak'in bu amac icin yerlesik gelen istemcisidir.
    data = {
        "grant_type": "password",
        "client_id": "admin-cli",
        "username": settings.keycloak.admin_user,
        "password": settings.keycloak.admin_password,
    }
    try:
        resp = requests.post(_admin_token_url(), data=data, timeout=10)
    except requests.RequestException as exc:
        raise KeycloakError(f"Keycloak admin endpoint unreachable: {exc}") from exc
    # 200 disinda ise genellikle yanlis admin kimlik bilgisi; net bir ipucu veriyoruz.
    if resp.status_code != 200:
        raise KeycloakError("Admin authorization failed (check KEYCLOAK_ADMIN credentials)")
    return resp.json()["access_token"]


def create_user(
    username: str,
    password: str,
    email: str,
    first_name: str = "",
    last_name: str = "",
    role: str = "student",
) -> None:
    """Create a Keycloak user and assign them a realm role.

    role must be one of: 'student', 'instructor'. Raises KeycloakError
    on conflict (username/email taken) or other Admin API failures.
    """
    # Once admin token alip yetkili istek basliklarini hazirlariz.
    admin_token = _get_admin_token()
    headers = {
        "Authorization": f"Bearer {admin_token}",
        "Content-Type": "application/json",
    }

    # Yeni kullanici govdesi. enabled/emailVerified=True: kullanici dogrudan giris yapabilsin.
    # temporary=False: parola kalici, ilk giriste sifre degistirme zorunlulugu yok.
    payload = {
        "username": username,
        "email": email,
        "firstName": first_name,
        "lastName": last_name,
        "enabled": True,
        "emailVerified": True,
        "credentials": [
            {"type": "password", "value": password, "temporary": False}
        ],
    }

    resp = requests.post(
        f"{_admin_realm_url()}/users",
        json=payload,
        headers=headers,
        timeout=15,
    )
    # 201 Created: kullanici olustu. Rol 'student' degilse ayrica realm rolu atariz
    # (student zaten realm varsayilani oldugu icin ayrica atamaya gerek yok).
    if resp.status_code == 201:
        if role and role != "student":  # student is realm default
            _assign_realm_role(username, role, admin_token)
        return
    # 409 Conflict: ayni kullanici adi ya da e-posta zaten kayitli.
    if resp.status_code == 409:
        raise KeycloakError("Username or email already in use")
    # Diger hatalarda Keycloak'in errorMessage alanini ayikla; JSON degilse ham metne dus.
    try:
        msg = resp.json().get("errorMessage", "Unknown registration error")
    except (json.JSONDecodeError, ValueError):
        msg = resp.text or "Unknown registration error"
    raise KeycloakError(f"Registration failed: {msg}")


def _assign_realm_role(username: str, role_name: str, admin_token: str) -> None:
    """Add a realm role to a user (used to promote a freshly-created user).

    Best-effort: if user or role lookup fails we log and move on so
    the user can still log in with their default role.
    """
    headers = {
        "Authorization": f"Bearer {admin_token}",
        "Content-Type": "application/json",
    }
    # Rol atama "best-effort"tur: bu blokta olusan ag hatalari (asagidaki except)
    # yutulup yalnizca loglanir; cunku rol atanamasa bile kullanici varsayilan
    # roluyle giris yapabilmeli, kayit islemi bu yuzden cokmemeli.
    try:
        # 1) Kullanici adindan kullanici id'sini bul. exact=true: kismi eslesmeleri engeller.
        users_resp = requests.get(
            f"{_admin_realm_url()}/users",
            params={"username": username, "exact": "true"},
            headers=headers,
            timeout=10,
        )
        users = users_resp.json() if users_resp.status_code == 200 else []
        if not users:
            log.warning("User %s not found for role assignment", username)
            return
        user_id = users[0]["id"]

        # 2) Atanacak rolun realm'deki tam tanimini (id dahil) cek.
        role_resp = requests.get(
            f"{_admin_realm_url()}/roles/{role_name}",
            headers=headers,
            timeout=10,
        )
        if role_resp.status_code != 200:
            log.warning("Role %s not found in realm", role_name)
            return
        role = role_resp.json()

        # 3) Rolu kullanicinin realm rol eslemelerine ekle (liste halinde beklenir).
        requests.post(
            f"{_admin_realm_url()}/users/{user_id}/role-mappings/realm",
            json=[role],
            headers=headers,
            timeout=10,
        )
    except requests.RequestException as exc:
        log.warning("Role assignment for %s failed: %s", username, exc)
