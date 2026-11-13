# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
import base64
import struct
import time

import jwt
import pytest
import strawberry
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from starlette.applications import Starlette
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from scheduler.auth import (
    Access,
    AuthError,
    AuthSettings,
    authenticate,
    bearer_token,
    load_public_key,
    verify_token,
)
from scheduler.graphql_mid.server import AuthenticatedGraphQL


@pytest.fixture(scope="module")
def private_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def other_private_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _standard(role: str) -> dict:
    return {
        "type": "standard",
        "id": "u-1",
        "role": {"type": role, "id": "r-1"},
        "otherRoles": [],
        "profile": {"orcidId": "0000-0001", "profile": {"givenName": "Ada", "familyName": "Lovelace"}},
    }


def _token(key, user: dict, **overrides) -> str:
    now = int(time.time())
    claims = {
        "iss": "lucuma-sso",
        "aud": ["lucuma"],
        "sub": user["id"],
        "iat": now - 10,
        "nbf": now - 10,
        "exp": now + 600,
        "lucuma-user": user,
    }
    claims.update(overrides)
    return jwt.encode(claims, key, algorithm="RS512")


def _mpi(n: int) -> bytes:
    return struct.pack(">H", n.bit_length()) + n.to_bytes((n.bit_length() + 7) // 8, "big")


def _pgp_armor(public_key) -> str:
    """A minimal ASCII-armored v4 RSA public key, as gpg --armor --export lays it out."""
    nums = public_key.public_numbers()
    body = bytes([4]) + struct.pack(">I", 1_700_000_000) + bytes([1]) + _mpi(nums.n) + _mpi(nums.e)
    # Old-format header, tag 6, two-byte length; then a user id packet (tag 13).
    packet = bytes([0x80 | (6 << 2) | 1]) + struct.pack(">H", len(body)) + body
    uid = b"lucuma-sso"
    packet += bytes([0xC0 | 13, len(uid)]) + uid
    b64 = base64.b64encode(packet).decode()
    lines = [b64[i : i + 64] for i in range(0, len(b64), 64)]
    return "\n".join(
        [
            "-----BEGIN PGP PUBLIC KEY BLOCK-----",
            "Version: test",
            "",
            *lines,
            "=abcd",
            "-----END PGP PUBLIC KEY BLOCK-----",
        ]
    )


def test_loads_pgp_armored_key(private_key):
    key = load_public_key(_pgp_armor(private_key.public_key()))
    assert key.public_numbers() == private_key.public_key().public_numbers()


def test_loads_pgp_key_with_escaped_newlines(private_key):
    key = load_public_key(_pgp_armor(private_key.public_key()).replace("\n", "\\n"))
    assert key.public_numbers() == private_key.public_key().public_numbers()


def test_loads_pem_key(private_key):
    pem = (
        private_key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    assert load_public_key(pem).public_numbers() == private_key.public_key().public_numbers()


@pytest.mark.parametrize(
    "value, expected",
    [
        ("Bearer abc", "abc"),
        ("bearer  abc ", "abc"),
        ("Basic abc", None),
        ("Bearer", None),
        (None, None),
        ("", None),
    ],
)
def test_bearer_token(value, expected):
    assert bearer_token(value) == expected


@pytest.mark.parametrize("role, access", [("staff", Access.STAFF), ("admin", Access.ADMIN)])
def test_staff_and_admin_pass(private_key, role, access):
    user = verify_token(_token(private_key, _standard(role)), private_key.public_key())
    assert user.access == access
    assert user.name == "Ada Lovelace"
    assert user.id == "u-1"


def test_service_user_passes(private_key):
    user = verify_token(
        _token(private_key, {"type": "service", "id": "u-9", "name": "navigate"}), private_key.public_key()
    )
    assert user.access == Access.SERVICE


@pytest.mark.parametrize("user", [_standard("pi"), _standard("ngo"), {"type": "guest", "id": "u-2"}])
def test_low_roles_are_forbidden(private_key, user):
    with pytest.raises(AuthError) as e:
        verify_token(_token(private_key, user), private_key.public_key())
    assert e.value.status_code == 403


@pytest.mark.parametrize(
    "overrides",
    [
        {"exp": int(time.time()) - 3600},
        {"aud": ["someone-else"]},
        {"iss": "not-sso"},
        {"lucuma-user": None},
    ],
)
def test_bad_claims_are_unauthorized(private_key, overrides):
    with pytest.raises(AuthError) as e:
        verify_token(_token(private_key, _standard("staff"), **overrides), private_key.public_key())
    assert e.value.status_code == 401


def test_wrong_signature_is_unauthorized(private_key, other_private_key):
    with pytest.raises(AuthError) as e:
        verify_token(_token(other_private_key, _standard("staff")), private_key.public_key())
    assert e.value.status_code == 401


def test_other_algorithms_are_refused(private_key):
    token = jwt.encode(
        {"iss": "lucuma-sso", "aud": "lucuma", "exp": int(time.time()) + 60, "lucuma-user": _standard("staff")},
        "secret",
        algorithm="HS512",
    )
    with pytest.raises(AuthError):
        verify_token(token, private_key.public_key())


def test_required_without_token_is_unauthorized(private_key):
    with pytest.raises(AuthError) as e:
        authenticate(None, AuthSettings(required=True, public_key=private_key.public_key()))
    assert e.value.status_code == 401


def test_required_without_key_refuses_everything(private_key):
    with pytest.raises(AuthError) as e:
        authenticate(f"Bearer {_token(private_key, _standard('staff'))}", AuthSettings(True, None))
    assert e.value.status_code == 503


def test_not_required_lets_anonymous_through_but_still_checks_tokens(private_key, other_private_key):
    settings = AuthSettings(required=False, public_key=private_key.public_key())
    assert authenticate(None, settings) is None
    with pytest.raises(AuthError):
        authenticate(f"Bearer {_token(other_private_key, _standard('staff'))}", settings)


# --- The GraphQL server -----------------------------------------------------


@strawberry.type
class _Query:
    @strawberry.field
    def whoami(self, info: strawberry.Info) -> str:
        user = info.context["user"]
        return user.name if user else "anonymous"


@strawberry.type
class _Subscription:
    @strawberry.subscription
    async def whoami(self, info: strawberry.Info) -> strawberry.scalars.JSON:
        user = info.context["user"]
        yield user.name if user else "anonymous"


def _client(settings: AuthSettings) -> TestClient:
    schema = strawberry.Schema(query=_Query, subscription=_Subscription)
    app = Starlette()
    app.add_route("/graphql", AuthenticatedGraphQL(schema, settings=settings))
    app.add_websocket_route("/graphql", AuthenticatedGraphQL(schema, settings=settings))
    return TestClient(app)


def test_http_without_token_is_401(private_key):
    response = _client(AuthSettings(True, private_key.public_key())).post("/graphql", json={"query": "{ whoami }"})
    assert response.status_code == 401


def test_http_with_pi_token_is_403(private_key):
    response = _client(AuthSettings(True, private_key.public_key())).post(
        "/graphql",
        json={"query": "{ whoami }"},
        headers={"Authorization": f"Bearer {_token(private_key, _standard('pi'))}"},
    )
    assert response.status_code == 403


def test_http_with_staff_token_reaches_the_resolver_with_the_user(private_key):
    response = _client(AuthSettings(True, private_key.public_key())).post(
        "/graphql",
        json={"query": "{ whoami }"},
        headers={"Authorization": f"Bearer {_token(private_key, _standard('staff'))}"},
    )
    assert response.status_code == 200
    assert response.json() == {"data": {"whoami": "Ada Lovelace"}}


def test_http_without_auth_required_is_anonymous(private_key):
    response = _client(AuthSettings(False, None)).post("/graphql", json={"query": "{ whoami }"})
    assert response.json() == {"data": {"whoami": "anonymous"}}


def _ws_whoami(client: TestClient, payload: dict):
    with client.websocket_connect("/graphql", subprotocols=["graphql-transport-ws"]) as ws:
        ws.send_json({"type": "connection_init", "payload": payload})
        ack = ws.receive_json()
        assert ack["type"] == "connection_ack"
        ws.send_json({"id": "1", "type": "subscribe", "payload": {"query": "subscription { whoami }"}})
        return ws.receive_json()["payload"]["data"]["whoami"]


def test_ws_with_staff_token_gets_the_user(private_key):
    client = _client(AuthSettings(True, private_key.public_key()))
    payload = {"Authorization": f"Bearer {_token(private_key, _standard('admin'))}"}
    assert _ws_whoami(client, payload) == "Ada Lovelace"


def test_ws_without_token_is_closed_with_4403(private_key):
    client = _client(AuthSettings(True, private_key.public_key()))
    with client.websocket_connect("/graphql", subprotocols=["graphql-transport-ws"]) as ws:
        ws.send_json({"type": "connection_init", "payload": {}})
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
    assert e.value.code == 4403
