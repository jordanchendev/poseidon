#!/usr/bin/env python3
"""Narrow stdio MCP evidence server for one 2330 research session."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dsh_2330_poc import (
    RevisionLedger,
    ValidationError,
    _parse_time,
    load_bundle,
    select_snapshot,
)

TOOLS = [
    {
        "name": "read_snapshot",
        "description": "Read the evidence visible to this session's selected 2330 snapshot.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "read_previous_revision",
        "description": "Read only the immutable previous revision linked to this session.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
]


def response(request_id: Any, result: Any = None, error: str | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    if error is None:
        message["result"] = result
    else:
        message["error"] = {"code": -32602, "message": error}
    return message


def tool_result(value: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}]}


def previous_for_session(snapshot: dict[str, Any], ledger: RevisionLedger, revision_id: str) -> dict[str, Any]:
    previous = ledger.read_revision(revision_id)
    if previous is None:
        raise ValidationError("previous revision is not present")
    if previous.get("as_of") is None or _parse_time(previous["as_of"], "previous.as_of") >= _parse_time(
        snapshot["as_of"], "snapshot.as_of"
    ):
        raise ValidationError("previous revision must be earlier than selected snapshot")
    previous.pop("observed_runtime", None)
    return previous


def serve(snapshot: dict[str, Any], previous: dict[str, Any] | None) -> None:
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValidationError("request must be an object")
            request_id, method = request.get("id"), request.get("method")
            if method == "initialize":
                result: Any = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "dsh-2330-poc", "version": "1"},
                }
            elif method == "notifications/initialized":
                continue
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                params = request.get("params")
                name = params.get("name") if isinstance(params, dict) else None
                arguments = params.get("arguments", {}) if isinstance(params, dict) else None
                if arguments not in ({}, None):
                    raise ValidationError("tool arguments are not allowed")
                if name == "read_snapshot":
                    result = tool_result(snapshot)
                elif name == "read_previous_revision":
                    result = tool_result(previous if previous is not None else {"previous_revision": None})
                else:
                    raise ValidationError("unknown read-only tool")
            else:
                raise ValidationError("unsupported method")
            print(json.dumps(response(request_id, result), ensure_ascii=False), flush=True)
        except (json.JSONDecodeError, ValidationError) as error:
            print(
                json.dumps(
                    response(
                        request.get("id") if "request" in locals() and isinstance(request, dict) else None,
                        error=str(error),
                    ),
                    ensure_ascii=False,
                ),
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=os.environ.get("DSH_POC_BUNDLE"))
    parser.add_argument("--snapshot", default=os.environ.get("DSH_POC_SNAPSHOT"))
    parser.add_argument("--ledger", default=os.environ.get("DSH_POC_LEDGER"))
    parser.add_argument("--previous-revision-id")
    parser.set_defaults(previous_revision_id=os.environ.get("DSH_POC_PREVIOUS_REVISION_ID"))
    args = parser.parse_args()
    if not all((args.bundle, args.snapshot, args.ledger)):
        raise SystemExit("bundle, snapshot, and ledger must be set by arguments or DSH_POC_* environment")
    snapshot = select_snapshot(load_bundle(Path(args.bundle)), args.snapshot)
    previous = None
    if args.previous_revision_id:
        previous = previous_for_session(snapshot, RevisionLedger(Path(args.ledger)), args.previous_revision_id)
    serve(snapshot, previous)


if __name__ == "__main__":
    main()
