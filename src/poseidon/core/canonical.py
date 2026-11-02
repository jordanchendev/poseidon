"""Canonical JSON and its SHA-256, the identity of every frozen payload.

These digests are written to the DB (manifests, intents, client order refs, review inputs) and feed the
advisory-lock keys, so the encoding below must never change.
"""

import hashlib
import json

from poseidon.core.errors import ValidationError


def canonical_json(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValidationError("payload must be finite JSON") from error


def content_sha256(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
