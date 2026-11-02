"""Fixtures shared by the research runner tests (helpers: testkit.py)."""

import hashlib
import json
import sys
from pathlib import Path

import assemble
import live_keys
import pytest
import runner
from testkit import (
    AGENDA_ID,
    AS_OF,
    BRIDGE_SCRIPT,
    DRAFT_KEY,
    EVENT_IDX,
    HTML,
    JEV_KEY,
    ROW,
    N,
    aaa,
    bbb,
    ccc,
    day,
    fake_bars,
    fake_twse,
)

WORKTREE = Path(__file__).resolve().parents[2]


@pytest.fixture()
def runner_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(WORKTREE / "src"))
    src = tmp_path / "web"
    src.mkdir()
    (src / "umc-3q26.htm").write_text(HTML, encoding="utf-8")
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    draft_env, jev_env = secrets / "draft.env", secrets / "jev.env"
    draft_env.write_text(f"{live_keys.DRAFT_KEY_VAR}={DRAFT_KEY}\n")
    jev_env.write_text(f'export {live_keys.JEV_KEY_VAR}="{JEV_KEY}"\n')
    draft_env.chmod(0o600)
    jev_env.chmod(0o600)
    bridge_py = tmp_path / "bridge.py"
    bridge_py.write_text(BRIDGE_SCRIPT)
    bridge_state = tmp_path / "bridge.json"
    bridge_state.write_text(json.dumps({"rows": [ROW]}))
    calls = {"fetch": []}

    def fetch(url, **kw):
        calls["fetch"].append(url)
        if "missing" in url:
            raise assemble.AssembleError(f"source fetch failed for {url}: HTTP 404")
        doc, stats = assemble.fetch_document((src / url.rsplit("/", 1)[-1]).as_uri(), **kw)
        doc["source_uri"] = url
        return doc, stats

    assemble_kw = {
        "fetch": fetch,
        "fetch_bars": fake_bars,
        "fetch_rows": fake_twse,
        "thalassa_base_url": "http://thalassa.test:8001",
    }
    state = tmp_path / "state"

    def argv(*extra, draft_env_path=draft_env):
        return [
            "--state-dir",
            str(state),
            "--draft-env",
            str(draft_env_path),
            "--jev-env",
            str(jev_env),
            "--bridge",
            f"{sys.executable} {bridge_py} {bridge_state}",
            *extra,
        ]

    def run(transport, *extra, kw=None, **akw):
        return runner.main(argv(*extra, **akw), transport=transport, assemble_kw=kw or assemble_kw, as_of=AS_OF)

    def bridge():
        return json.loads(bridge_state.read_text())

    def set_bridge(**upd):
        st = bridge()
        st.update(upd)
        bridge_state.write_text(json.dumps(st))

    return dict(
        bridge_cmd=f"{sys.executable} {bridge_py} {bridge_state}",
        tmp=tmp_path,
        state=state,
        run=run,
        bridge=bridge,
        set_bridge=set_bridge,
        assemble_kw=assemble_kw,
        calls=calls,
        draft_env=draft_env,
        run_dir=state / "runs" / AGENDA_ID,
    )


@pytest.fixture()
def theme_env(tmp_path):
    csv_path = tmp_path / "bars.csv"
    rows = []
    for sym, f in (("AAA", aaa), ("BBB", bbb), ("CCC", ccc)):
        for i in range(N):
            c = f(i)
            rows.append(f"{day(i)}T00:00:00Z,{sym},{c - 0.5},{c + 1},{c - 1},{c},{1000 + i}")
    csv_path.write_text("\n".join(rows) + "\n")
    raw = tmp_path / "raw.txt"
    text = "Company announced record revenue of 100 units for the quarter, up 20% year over year."
    raw.write_text(text + "\n")
    cfg = {
        "scope_key": "tw_stock:spot:AAA",
        "symbol": "AAA",
        "as_of": f"{day(N - 1)}T09:00:00Z",
        "event_time": f"{day(EVENT_IDX)}T10:00:00Z",
        "account_scope": "paper:test",
        "universe_id": "u-test",
        "thalassa_base_url": "http://127.0.0.1:1",
        "symbols": ["AAA", "BBB"],
        "benchmark": "BBB",
        "required_data": {"fundamental": {"max_age_seconds": 2592000}, "ohlcv": {"max_age_seconds": 172800}},
        "policy_version": "test/v0",
        "provider": "test",
        "model": "test",
        "documents": [
            {
                "id": "fundamental:doc1",
                "kind": "fundamental",
                "title": "Doc",
                "publisher": "Pub",
                "source_uri": "https://example.com/doc1",
                "event_time": f"{day(EVENT_IDX)}T10:00:00Z",
                "available_at": f"{day(EVENT_IDX)}T10:00:00Z",
                "raw_text_path": str(raw),
                "raw_files": {"raw.txt": hashlib.sha256(raw.read_bytes()).hexdigest()},
                "excerpts": [{"locator": "line 1", "text": text}],
            }
        ],
    }
    narrative = {
        "thesis": "Test thesis.",
        "stance": "watch",
        "change_summary": "First version.",
        "next_review_at": "2026-10-05T01:00:00Z review the October revenue",
        "tl_dr": "AAA beat and outperformed the benchmark.",
        "risks": ["r"],
        "invalidation_conditions": ["i"],
        "unknowns": ["u"],
        "claims": [
            {"kind": "fact", "text": "Revenue was 100 units.", "evidence_ids": ["fundamental:doc1"]},
            {
                "kind": "fact",
                "text": "AAA outperformed BBB after the event.",
                "evidence_ids": ["price_facts:AAA", "ohlcv:AAA"],
            },
            {
                "kind": "inference",
                "text": "The move is probably not fully priced in.",
                "evidence_ids": ["price_facts:AAA"],
            },
        ],
        "catalysts": [
            {"date": "2026-09-10", "type": "other", "expected_signal": "past, ignored", "source": "s"},
            {
                "date": "2026-10-05",
                "type": "monthly_revenue",
                "expected_signal": "revenue stays above 100",
                "source": "MOPS",
            },
            {"date": "2026-11-12", "type": "earnings_call", "expected_signal": "guidance", "source": "IR"},
        ],
    }
    return dict(tmp=tmp_path, csv=csv_path, cfg=cfg, narrative=narrative)
