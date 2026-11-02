"""Fixtures shared by the research runner tests (helpers: testkit.py)."""

import json
import sys
from pathlib import Path

import assemble
import draft
import pytest
import runner
from testkit import AGENDA_ID, AS_OF, BRIDGE_SCRIPT, DRAFT_KEY, HTML, JEV_KEY, ROW, fake_bars, fake_twse

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
    draft_env.write_text(f"{draft.DRAFT_KEY_VAR}={DRAFT_KEY}\n")
    jev_env.write_text(f'export {runner.JEV_KEY_VAR}="{JEV_KEY}"\n')
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
