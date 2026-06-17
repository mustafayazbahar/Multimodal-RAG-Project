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

# auto_error=False: token yoksa otomatik 403 firlatma; eksik token durumunu
# asagida kendimiz 401 olarak ele aliyoruz (daha tutarli hata mesaji icin).
_security = HTTPBearer(auto_error=False)


# Dogrulanmis kullaniciyi temsil eden basit veri tasiyici (kullanici adi + rol).
@dataclass
class CurrentUser:
    username: str
    role: str


# Korumali endpoint'lerde Depends ile kullanilan ana kimlik dogrulama bagimliligi.
# Authorization: Bearer ... basligindaki JWT'yi Keycloak'a karsi dogrular ve
# claim'lerden kullanici bilgisini cikarir.
def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_security)],
) -> CurrentUser:
    # Token hic gonderilmemisse istegi 401 ile reddet.
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token"
        )
    # Imza/son kullanma gibi dogrulama hatalarinda KeycloakError'i 401'e cevir.
    try:
        claims = verify_token(credentials.credentials)
    except KeycloakError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)
        ) from exc
    # Dogrulanan claim'lerden preferred_username ve realm rolu cekiliyor.
    user = extract_user(claims)
    return CurrentUser(username=user["username"], role=user["role"])


# get_current_user uzerine ince bir sarmalayici: sadece "instructor" rolune
# sahip kullanicilara izin verir. Yikici islemlerde (PDF yukleme, ingestion
# calistirma, bilgi tabanini sifirlama) yetki kapisi olarak kullanilir.
def require_instructor(
    user: Annotated[CurrentUser, Depends(get_current_user)],
) -> CurrentUser:
    # Rol instructor degilse 403 ile eris engellenir.
    if user.role != "instructor":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Instructor role required"
        )
    return user
