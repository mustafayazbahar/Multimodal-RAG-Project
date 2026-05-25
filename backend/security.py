"""FastAPI auth dependency — validates Keycloak-signed JWTs.

Replaces the legacy HS256 self-signed flow. Every protected endpoint
takes `Depends(get_current_user)`, which:
  1. Reads `Authorization: Bearer ...` off the request.
  2. Verifies the JWT against Keycloak's published JWKS (signature,
     expiry).
  3. Pulls `preferred_username` + realm role out of the claims.

`require_instructor` is a thin wrapper that rejects non-instructor
users with 403, used by destructive endpoints (upload, run ingestion,
reset KB).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from services.keycloak_auth import KeycloakError, extract_user, verify_token

_security = HTTPBearer(auto_error=False)


@dataclass
class CurrentUser:
    username: str
    role: str


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_security)],
) -> CurrentUser:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token"
        )
    try:
        claims = verify_token(credentials.credentials)
    except KeycloakError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)
        ) from exc
    user = extract_user(claims)
    return CurrentUser(username=user["username"], role=user["role"])


def require_instructor(
    user: Annotated[CurrentUser, Depends(get_current_user)],
) -> CurrentUser:
    if user.role != "instructor":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Instructor role required"
        )
    return user
