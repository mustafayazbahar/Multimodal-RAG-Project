"""Auth endpoints: register / login — both proxy to Keycloak.

Login is a Keycloak password-grant call; the token we ship to the
frontend is the actual Keycloak access_token (signed RS256 by the
realm), so the frontend can stash it in localStorage and we can
re-verify on every subsequent request without round-tripping to
Keycloak again.

Register hits the Keycloak Admin API with the master-realm admin
credentials configured via KEYCLOAK_ADMIN / KEYCLOAK_ADMIN_PASSWORD.
After a successful registration we auto-login the new user so the
client gets a token immediately and the UX matches the old flow.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.schemas import LoginRequest, RegisterRequest, TokenResponse
from services.keycloak_auth import (
    KeycloakError,
    create_user,
    extract_user,
    login as kc_login,
    verify_token,
)

router = APIRouter(prefix="/auth", tags=["auth"])


def _token_response(access_token: str) -> TokenResponse:
    claims = verify_token(access_token)
    user = extract_user(claims)
    return TokenResponse(
        access_token=access_token,
        role=user["role"],
        username=user["username"],
    )


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest) -> TokenResponse:
    try:
        token_data = kc_login(payload.username.strip(), payload.password)
    except KeycloakError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)
        ) from exc
    return _token_response(token_data["access_token"])


@router.post("/register", response_model=TokenResponse)
def register(payload: RegisterRequest) -> TokenResponse:
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
    try:
        token_data = kc_login(payload.username.strip(), payload.password)
    except KeycloakError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Registered but auto-login failed: {exc}",
        ) from exc
    return _token_response(token_data["access_token"])
