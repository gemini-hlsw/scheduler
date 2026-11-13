# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""Verification of GPP SSO (lucuma-sso) JWTs.

The UI signs in through SSO and sends the JWT as ``Authorization: Bearer <jwt>``
on HTTP requests and in the graphql-ws ``connection_init`` payload. SSO signs
with RS512, issuer ``lucuma-sso`` and audience ``lucuma``; the signed-in user is
in the ``lucuma-user`` claim (see lucuma-odb ``modules/sso-service``).

SSO does not serve its public key: like lucuma-odb (``ODB_SSO_PUBLIC_KEY``), the
service is given it in ``SSO_PUBLIC_KEY``, as the ASCII-armored PGP key SSO is
configured with, or as a PEM ``PUBLIC KEY``.

Settings (environment):
    SSO_AUTH_REQUIRED  ``true`` (default) or ``false``. When false, requests are
                       let through without a token (local development, scripts).
    SSO_PUBLIC_KEY     The SSO public key. Required when auth is required; without
                       it every request is refused, never let through.
"""

from __future__ import annotations

import base64
import os
import struct
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from typing import Any, Mapping, Optional

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey, RSAPublicNumbers

from scheduler.services.logger_factory import create_logger

_logger = create_logger(__name__)

ISSUER = "lucuma-sso"
AUDIENCE = "lucuma"
ALGORITHM = "RS512"
# Tolerated clock skew between SSO and this server, in seconds.
LEEWAY_SECONDS = 30


class Access(IntEnum):
    """lucuma role hierarchy, in increasing power."""

    GUEST = 0
    PI = 1
    NGO = 2
    STAFF = 3
    ADMIN = 4
    SERVICE = 5


# The lowest role that may use the scheduler, matching the UI's REQUIRED_ACCESS.
REQUIRED_ACCESS = Access.STAFF


class AuthError(Exception):
    """The request carries no usable token (401) or a user without access (403)."""

    def __init__(self, message: str, status_code: int = 401):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class SsoUser:
    """The user a verified token was issued to."""

    id: str
    type: str  # standard | guest | service
    access: Access
    name: str
    claims: Mapping[str, Any]

    @classmethod
    def from_lucuma_user(cls, user: Mapping[str, Any]) -> SsoUser:
        user_type = user.get("type")
        user_id = str(user.get("id", ""))
        if user_type == "guest":
            return cls(user_id, "guest", Access.GUEST, "Guest User", user)
        if user_type == "service":
            return cls(user_id, "service", Access.SERVICE, f"Service user ({user.get('name', '')})", user)
        if user_type == "standard":
            role = (user.get("role") or {}).get("type", "")
            try:
                access = Access[role.upper()]
            except KeyError:
                raise AuthError(f"Unknown role {role!r} in token.") from None
            return cls(user_id, "standard", access, _display_name(user), user)
        raise AuthError(f"Unknown user type {user_type!r} in token.")

    def __str__(self) -> str:
        return f"{self.name} [{self.id}, {self.access.name.lower()}]"


def _display_name(user: Mapping[str, Any]) -> str:
    orcid = user.get("profile") or {}
    p = orcid.get("profile") or {}
    if p.get("creditName"):
        return p["creditName"]
    full = " ".join(n for n in (p.get("givenName"), p.get("familyName")) if n)
    return full or orcid.get("orcidId") or str(user.get("id", ""))


# --- Public key -------------------------------------------------------------


def load_public_key(text: str) -> RSAPublicKey:
    """The RSA key from a PEM public key or an ASCII-armored PGP public key."""
    # Config UIs sometimes store a multi-line value with literal "\n".
    text = text.strip().replace("\\n", "\n")
    if "BEGIN PGP PUBLIC KEY BLOCK" in text:
        return _rsa_from_pgp(_dearmor(text))
    key = serialization.load_pem_public_key(text.encode())
    if not isinstance(key, RSAPublicKey):
        raise ValueError("SSO public key is not an RSA key.")
    return key


def _dearmor(text: str) -> bytes:
    """The binary packets inside an ASCII-armored PGP block (RFC 4880 §6.2)."""
    lines = [line.strip() for line in text.splitlines()]
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith("-----BEGIN PGP"))
        end = next(i for i, line in enumerate(lines) if line.startswith("-----END PGP"))
    except StopIteration:
        raise ValueError("Malformed PGP armor.") from None
    body = lines[start + 1 : end]
    # Armor headers (e.g. "Version: …") end at the first blank line, if any.
    if "" in body and any(":" in line for line in body[: body.index("")]):
        body = body[body.index("") + 1 :]
    data = "".join(line for line in body if line and not line.startswith("="))  # "=" starts the CRC
    return base64.b64decode(data)


def _rsa_from_pgp(data: bytes) -> RSAPublicKey:
    """The first RSA public-key packet of a PGP key (RFC 4880 §4.2, §5.5.2)."""
    i = 0
    while i < len(data):
        header = data[i]
        if not header & 0x80:
            raise ValueError("Malformed PGP packet header.")
        if header & 0x40:  # new format
            tag = header & 0x3F
            first = data[i + 1]
            if first < 192:
                length, i = first, i + 2
            elif first < 224:
                length, i = ((first - 192) << 8) + data[i + 2] + 192, i + 3
            elif first == 255:
                length, i = struct.unpack(">I", data[i + 2 : i + 6])[0], i + 6
            else:
                raise ValueError("Partial-length PGP packets are not supported in keys.")
        else:  # old format
            tag = (header >> 2) & 0x0F
            size = {0: 1, 1: 2, 2: 4}.get(header & 0x03)
            if size is None:
                raise ValueError("Indeterminate-length PGP packets are not supported.")
            length = int.from_bytes(data[i + 1 : i + 1 + size], "big")
            i += 1 + size
        body = data[i : i + length]
        i += length
        if tag == 6:  # public key
            if body[0] != 4:
                raise ValueError(f"PGP key version {body[0]} is not supported (expected 4).")
            if body[5] not in (1, 2, 3):
                raise ValueError("SSO PGP key is not an RSA key.")
            n, offset = _mpi(body, 6)
            e, _ = _mpi(body, offset)
            return RSAPublicNumbers(e, n).public_key()
    raise ValueError("No public-key packet in the PGP key.")


def _mpi(data: bytes, offset: int) -> tuple[int, int]:
    bits = struct.unpack(">H", data[offset : offset + 2])[0]
    size = (bits + 7) // 8
    start = offset + 2
    return int.from_bytes(data[start : start + size], "big"), start + size


# --- Verification -----------------------------------------------------------


@dataclass(frozen=True)
class AuthSettings:
    required: bool
    public_key: Optional[RSAPublicKey]


@lru_cache(maxsize=1)
def auth_settings() -> AuthSettings:
    required = os.environ.get("SSO_AUTH_REQUIRED", "true").strip().lower() not in ("false", "0", "no")
    raw_key = os.environ.get("SSO_PUBLIC_KEY", "").strip()
    public_key = None
    if raw_key:
        try:
            public_key = load_public_key(raw_key)
        except Exception as e:
            _logger.error(f"SSO_PUBLIC_KEY could not be read; every authenticated request will be refused: {e}")
    if not required:
        _logger.warning("SSO_AUTH_REQUIRED=false: the GraphQL API accepts requests without a token.")
    elif public_key is None:
        _logger.error("SSO auth is required but SSO_PUBLIC_KEY is not set; every request will be refused.")
    return AuthSettings(required, public_key)


def bearer_token(authorization: Optional[str]) -> Optional[str]:
    """The token in an ``Authorization: Bearer <token>`` value, if any."""
    if not authorization:
        return None
    scheme, _, token = authorization.strip().partition(" ")
    if scheme.lower() != "bearer":
        return None
    return token.strip() or None


def verify_token(token: str, public_key: RSAPublicKey) -> SsoUser:
    """The user of a valid SSO token. Raises AuthError (401) when the token is
    invalid or expired, and AuthError (403) when its role is too low."""
    try:
        claims = jwt.decode(
            token,
            public_key,
            algorithms=[ALGORITHM],
            audience=AUDIENCE,
            issuer=ISSUER,
            leeway=LEEWAY_SECONDS,
            options={"require": ["exp", "iss", "aud"]},
        )
    except jwt.ExpiredSignatureError:
        raise AuthError("Token expired.") from None
    except jwt.InvalidTokenError as e:
        raise AuthError(f"Invalid token: {e}") from None
    lucuma_user = claims.get("lucuma-user")
    if not isinstance(lucuma_user, Mapping):
        raise AuthError("Token carries no lucuma-user.")
    user = SsoUser.from_lucuma_user(lucuma_user)
    if user.access < REQUIRED_ACCESS:
        raise AuthError(
            f"The scheduler needs a {REQUIRED_ACCESS.name.lower()} role or higher; "
            f"the token's role is {user.access.name.lower()}.",
            status_code=403,
        )
    return user


def authenticate(authorization: Optional[str], settings: Optional[AuthSettings] = None) -> Optional[SsoUser]:
    """The user behind an Authorization value.

    Returns None only when auth is not required and no valid token was sent.
    Raises AuthError otherwise. With auth not required, a token that is sent is
    still verified when the key is known, so a bad token is never ignored.
    """
    settings = settings or auth_settings()
    token = bearer_token(authorization)
    if token is None:
        if settings.required:
            raise AuthError("Missing bearer token.")
        return None
    if settings.public_key is None:
        if settings.required:
            raise AuthError("Server is not configured to verify tokens.", status_code=503)
        return None
    return verify_token(token, settings.public_key)
