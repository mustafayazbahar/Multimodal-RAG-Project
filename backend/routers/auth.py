"""Auth endpoints: register / login."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.schemas import LoginRequest, RegisterRequest, TokenResponse
from backend.security import create_token
from services.auth import login_user, register_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse)
def register(payload: RegisterRequest) -> TokenResponse:
    if not register_user(payload.username, payload.password):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username taken or invalid input",
        )
    token = create_token(payload.username.strip(), "student")
    return TokenResponse(access_token=token, role="student", username=payload.username.strip())


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest) -> TokenResponse:
    ok, role = login_user(payload.username, payload.password)
    if not ok or role is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )
    token = create_token(payload.username.strip(), role)
    return TokenResponse(access_token=token, role=role, username=payload.username.strip())
