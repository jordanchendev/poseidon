"""Credential-free principals and fail-closed decision-loop authorization."""

import asyncio
import hashlib
import json

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from poseidon.api.auth import get_auth_principal, verify_api_key
from poseidon.core.config import settings


@pytest.fixture(autouse=True)
def auth_config(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "legacy-secret")
    monkeypatch.setattr(settings, "api_principals_json", "{}")


def configure(key="human-secret", **overrides):
    record = {
        "actor_id": "human:operator",
        "roles": ["researcher", "portfolio_manager"],
        "account_scopes": ["paper:pilot"],
    }
    record.update(overrides)
    settings.api_principals_json = json.dumps({hashlib.sha256(key.encode()).hexdigest(): record})


def resolve(key="human-secret"):
    return asyncio.run(get_auth_principal(key))


def test_mapping_has_stable_identity_without_credential_material():
    configure()
    principal = resolve()
    assert principal.actor_id == "human:operator"
    assert principal.roles == frozenset({"researcher", "portfolio_manager"})
    assert principal.account_scopes == frozenset({"paper:pilot"})
    principal.require_role("portfolio_manager")
    principal.require_account_scope("paper:pilot")
    assert "human-secret" not in repr(principal)
    assert hashlib.sha256(b"human-secret").hexdigest() not in repr(principal)
    configure(key="rotated-secret")
    assert resolve("rotated-secret") == principal


def test_wrong_role_and_cross_account_are_forbidden():
    configure()
    principal = resolve()
    for check, value in (
        (principal.require_role, "release_manager"),
        (principal.require_account_scope, "paper:other"),
    ):
        with pytest.raises(HTTPException) as error:
            check(value)
        assert error.value.status_code == 403


def test_worker_cannot_submit_research_approve_or_release():
    configure(actor_id="service:decision-worker", roles=["decision-worker"])
    principal = resolve()
    for role in ("researcher", "portfolio_manager", "release_manager"):
        with pytest.raises(HTTPException) as error:
            principal.require_role(role)
        assert error.value.status_code == 403


def test_legacy_key_keeps_secured_endpoint_access_without_authority():
    configure(key="legacy-secret")
    app = FastAPI()

    @app.get("/secured", dependencies=[Depends(verify_api_key)])
    def secured():
        return {"ok": True}

    client = TestClient(app)
    assert client.get("/secured", headers={"X-API-Key": "legacy-secret"}).status_code == 200
    assert client.get("/secured", headers={"X-API-Key": "wrong"}).status_code == 401
    principal = resolve("legacy-secret")
    assert principal.actor_id == "legacy-api-key"
    assert not principal.roles and not principal.account_scopes
    for role in ("portfolio_manager", "release_manager"):
        with pytest.raises(HTTPException):
            principal.require_role(role)


def test_mapped_key_cannot_inherit_legacy_route_authority():
    configure()
    with pytest.raises(HTTPException) as error:
        asyncio.run(verify_api_key("human-secret"))
    assert error.value.status_code == 401


@pytest.mark.parametrize(
    "config", ["not json", "[]", '{"not-a-fingerprint": {}}', '{"duplicate": {}, "duplicate": {}}']
)
def test_invalid_json_configuration_fails_closed(config):
    settings.api_principals_json = config
    with pytest.raises(HTTPException) as error:
        resolve("legacy-secret")
    assert error.value.status_code == 401


@pytest.mark.parametrize(
    "overrides",
    [
        {"actor_id": ""},
        {"roles": "portfolio_manager"},
        {"roles": ["admin"]},
        {"account_scopes": "*"},
        {"account_scopes": ["*"]},
        {"account_scopes": [""]},
        {"roles": ["decision-worker", "portfolio_manager"]},
        {"api_key": "raw-secret"},
    ],
)
def test_malformed_principal_configuration_fails_closed(overrides):
    configure(**overrides)
    with pytest.raises(HTTPException) as error:
        resolve()
    assert error.value.status_code == 401


def test_unmapped_key_is_rejected():
    configure()
    with pytest.raises(HTTPException) as error:
        resolve("unknown-secret")
    assert error.value.status_code == 401
