"""JWT-based session token + FastAPI dependencies for auth/role checks."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from services.config import settings

_ALGORITHM = "HS256"
_security = HTTPBearer(auto_error=False)


@dataclass
class CurrentUser:
    username: str
    role: str


def create_token(username: str, role: str) -> str:
    now = int(time.time())
    payload = {
        "sub": username,
        "role": role,
        "iat": now,
        "exp": now + settings.auth.jwt_ttl_hours * 3600,
    }
    return jwt.encode(payload, settings.auth.jwt_secret, algorithm=_ALGORITHM)


def _decode(token: str) -> dict:
    try:
        return jwt.decode(token, settings.auth.jwt_secret, algorithms=[_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_security)],
) -> CurrentUser:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    payload = _decode(credentials.credentials)
    return CurrentUser(username=payload["sub"], role=payload["role"])


def require_instructor(user: Annotated[CurrentUser, Depends(get_current_user)]) -> CurrentUser:
    if user.role != "instructor":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Instructor role required")
    return user
