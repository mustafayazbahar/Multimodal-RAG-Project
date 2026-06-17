"""Auth endpoints: register / login (password-grant) + OAuth Code flow proxy.

Two parallel login paths are exposed:

1. **Password grant** (`POST /auth/login`) — the legacy form-based path.
   Kept for API clients, tests, and automation. The Streamlit UI no
   longer surfaces it as the default but the endpoint still works.

2. **Authorization Code** (`GET /auth/login-url` →
   `POST /auth/exchange-code`) — the new browser-initiated path. The
   frontend redirects the user to Keycloak's `/auth` page; Keycloak
   sends them back with a `?code=` which the frontend exchanges via
   `/auth/exchange-code` to receive the same `TokenResponse` shape.

Register still hits the Keycloak Admin API and (for compatibility with
the old form UX) auto-logs the user in via the password grant.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from backend.schemas import (
    ExchangeCodeRequest,
    LoginRequest,
    OauthUrlResponse,
    RegisterRequest,
    TokenResponse,
)
from services.keycloak_auth import (
    KeycloakError,
    build_login_url,
    build_logout_url,
    create_user,
    exchange_code,
    extract_user,
    login as kc_login,
    verify_token,
)

# Bu modulun tum endpoint'leri /auth on eki altinda toplanir.
router = APIRouter(prefix="/auth", tags=["auth"])


# Yardimci: ham access_token'i dogrulayip claim'lerden rol/kullanici cikararak
# standart TokenResponse'a cevirir. Tum giris yollari (login/register/exchange)
# cevabi ayni sekle getirmek icin bunu kullanir.
def _token_response(access_token: str, id_token: str | None = None) -> TokenResponse:
    claims = verify_token(access_token)
    user = extract_user(claims)
    return TokenResponse(
        access_token=access_token,
        id_token=id_token,
        role=user["role"],
        username=user["username"],
    )


# Parola grant'i ile giris: kullanici adi/parola alir, Keycloak'tan token ister.
@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest) -> TokenResponse:
    # Kullanici adindaki bas/son bosluklar temizlenir; hatali kimlik 401 doner.
    try:
        token_data = kc_login(payload.username.strip(), payload.password)
    except KeycloakError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)
        ) from exc
    return _token_response(token_data["access_token"], token_data.get("id_token"))


# Yeni kullanici kaydi: Keycloak Admin API uzerinden hesap olusturur ve
# ardindan eski form akisiyla uyumlu olmak icin kullaniciyi otomatik giris yaptirir.
@router.post("/register", response_model=TokenResponse)
def register(payload: RegisterRequest) -> TokenResponse:
    # Yeni kayitlar her zaman "student" rolu ile acilir. Kullanici adi/email
    # zaten varsa Keycloak hata verir; bunu 409 Conflict olarak doneriz.
    try:
        create_user(
            username=payload.username.strip(),
            password=payload.password,
            email=payload.email.strip(),
            first_name=(payload.first_name or "").strip(),
            last_name=(payload.last_name or "").strip(),
            role="student",
        )
    except KeycloakError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc

    # Auto-login the freshly created user so the frontend gets a token
    # without a second round-trip.
    # Yeni olusturulan kullaniciyi hemen giris yaptir ki frontend ikinci bir
    # istek atmadan token alsin. Hesap acildi ama giris basarisiz olursa, durumu
    # acikca belirten bir 401 doneriz (kayit kismen tamamlandi).
    try:
        token_data = kc_login(payload.username.strip(), payload.password)
    except KeycloakError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Registered but auto-login failed: {exc}",
        ) from exc
    return _token_response(token_data["access_token"], token_data.get("id_token"))


# ─────────────────────────────────────────────────────────────────────────
# OAuth Authorization Code flow
# ─────────────────────────────────────────────────────────────────────────
# Adim 1: Frontend'in tarayiciyi yonlendirecegi Keycloak /auth URL'ini uretir.
# redirect_uri, Keycloak'in kullaniciyi kod ile geri gonderecegi adrestir.
@router.get("/login-url", response_model=OauthUrlResponse)
def get_login_url(redirect_uri: str = Query(..., description="Browser callback URL")) -> OauthUrlResponse:
    """Return the Keycloak `/auth` URL the frontend should redirect to."""
    return OauthUrlResponse(url=build_login_url(redirect_uri))


# Adim 2: Keycloak'tan donen yetki kodunu (code) gercek token'lara cevirir ve
# standart TokenResponse doner. Gecersiz kod/redirect durumunda 401 verir.
@router.post("/exchange-code", response_model=TokenResponse)
def post_exchange_code(payload: ExchangeCodeRequest) -> TokenResponse:
    """Exchange a Keycloak callback `code` for our standard TokenResponse."""
    try:
        token_data = exchange_code(payload.code, payload.redirect_uri)
    except KeycloakError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)
        ) from exc
    return _token_response(token_data["access_token"], token_data.get("id_token"))


# Cikis: Keycloak'in end-session URL'ini uretir. id_token_hint verilirse onay
# ekrani atlanir (sessiz cikis); bu yuzden frontend son id_token'i buraya iletir.
@router.get("/logout-url", response_model=OauthUrlResponse)
def get_logout_url(
    redirect_uri: str = Query(..., description="Where to send the user after logout"),
    id_token_hint: str | None = Query(None, description="Last id_token, enables silent logout"),
) -> OauthUrlResponse:
    """Return the Keycloak end-session URL so the frontend can navigate to it."""
    return OauthUrlResponse(url=build_logout_url(redirect_uri, id_token_hint))
