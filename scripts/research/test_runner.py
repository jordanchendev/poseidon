"""Offline tests for assemble.py / draft.py / runner.py.

No network: the source document is served from a local file:// URL through the real fetch_source code path, Thalassa
bars and TWSE rows come from fakes, OpenRouter and JEV go through an injected transport, and the agenda bridge is a
small Python script driven by a JSON state file. research_theme.py runs for real against a SQLite ledger.
Run from the Poseidon repo root:
  PYTHONPATH=src python -m pytest -q -p no:cacheprovider scripts/research/test_runner.py
"""

import datetime
import json
import sqlite3
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]
sys.path.insert(0, str(HERE))

import assemble  # noqa: E402
import draft  # noqa: E402
import runner  # noqa: E402

AGENDA_ID = "20261021-3"
DOC_ID = f"fundamental:agenda-{AGENDA_ID}"
AS_OF = datetime.datetime(2026, 10, 21, 4, 20, tzinfo=datetime.UTC)
DRAFT_KEY = "sk-or-test-draft-0000"
JEV_KEY = "sk-or-test-jev-1111"
RETURNED_MODEL = "deepseek/deepseek-v4.1-flash-1017"
HTML = """<html><body>
<p>UMC Reports Third Quarter 2026 Results. Revenue was NT$68.73 billion for the quarter, up 12.6% quarter over quarter.</p>

<p>Gross margin was 32.5% for the quarter and capacity utilization rate was 85% across the fabs this quarter.</p>

<p>The company expects wafer shipments to increase by high-single digit in the fourth quarter of 2026, per its outlook.</p>
</body></html>
"""

ROW = {
    "id": AGENDA_ID,
    "topic": "UMC 3Q26 results",
    "symbols": ["2303"],
    "event_date": "2026-10-06",
    "event_type": "earnings",
    "primary_source_url": "https://ir.example.test/umc-3q26.htm",
    "mirror_url": None,
    "why_now": "results published",
    "status": "accepted",
}


def good_narrative():
    return {
        "tl_dr": "聯電第三季營收與毛利率穩健，股價走勢仍待法說確認。",
        "thesis": "聯電第三季營收 NT$68.73 billion，毛利率 32.5%。",
        "stance": "watch",
        "change_summary": "首版。",
        "next_review_at": "2026-10-28T09:00:00Z 法說會後補版",
        "catalysts": [
            {"date": "2026-10-28", "type": "earnings_call", "expected_signal": "第四季指引", "source": DOC_ID}
        ],
        "risks": ["證據只有新聞稿。"],
        "invalidation_conditions": ["第四季出貨衰退。"],
        "unknowns": ["第四季毛利率。"],
        "claims": [
            {
                "kind": "fact",
                "text": "聯電第三季營收為 NT$68.73 billion。",
                "evidence_ids": [DOC_ID],
                "quote": "Revenue was NT$68.73 billion for the quarter",
            },
            {
                "kind": "fact",
                "text": "聯電第三季毛利率為 32.5%。",
                "evidence_ids": [DOC_ID],
                "quote": "Gross margin was 32.5% for the quarter",
            },
            {
                "kind": "inference",
                "text": "2303 的價格事實已凍結供事件後追蹤。",
                "evidence_ids": ["ohlcv:2303", "price_facts:2303"],
                "quote": "2303 price facts (algorithm price-facts-v2",
            },
        ],
    }


def bad_number_narrative():
    n = good_narrative()
    n["claims"][1]["text"] = "聯電第三季毛利率為 35.2%。"
    return n


# ------------------------------------------------------------------------------------------------ fakes
def fake_bars(symbol, start, end):
    base = {"2303": 150.0, "0050": 110.0}[symbol]
    out, d, i = [], datetime.date(2026, 7, 1), 0
    while d <= end.date():
        if d.weekday() < 5:
            ts = datetime.datetime(d.year, d.month, d.day, 5, 30, tzinfo=datetime.UTC)
            if ts <= end:
                c = base + (i % 7) - 3 + i * 0.1
                out.append((ts.isoformat(), c - 0.5, c + 1, c - 1, c, 1000.0 + i))
                i += 1
        d += datetime.timedelta(days=1)
    return out


def fake_twse():
    rows = [
        {
            "出表日期": "1151012",
            "資料年月": "11509",
            "公司代號": "2303",
            "公司名稱": "聯電",
            "營業收入-當月營收": "25350431",
            "營業收入-上月營收": "23180000",
            "營業收入-去年當月營收": "19925000",
            "營業收入-上月比較增減(%)": "9.36",
            "營業收入-去年同月增減(%)": "27.22",
        },
        {"出表日期": "1151012", "資料年月": "11509", "公司代號": "2330", "營業收入-當月營收": "1"},
    ]
    return rows, "https://openapi.twse.com.tw/v1/opendata/t187ap05_L", "2026-10-21T04:00:00Z"


class FakeOpenRouter:
    """Routes chat completions to a queue of draft responses and JEV calls to a label function."""

    def __init__(self, drafts, jev_label=lambda claim: "supports"):
        self.drafts = list(drafts)
        self.jev_label = jev_label
        self.chat_calls, self.jev_calls, self.headers_seen = [], [], []

    def __call__(self, url, headers, body, timeout):
        self.headers_seen.append(headers)
        req = json.loads(body)
        if url.endswith("/chat/completions"):
            self.chat_calls.append(req)
            item = self.drafts.pop(0)
            if isinstance(item, tuple):  # (status, raw body)
                return item
            if isinstance(item, dict) and "choices" in item:
                return 200, json.dumps(item).encode()
            return 200, json.dumps(chat_response(json.dumps(item, ensure_ascii=False))).encode()
        self.jev_calls.append(req)
        label = self.jev_label(req["state"]["claim"])
        return 200, json.dumps(
            {"answers": {"citation_relation": {"choice": label, "confidence": 0.97}}, "usage": {"cost": 0.0001}}
        ).encode()


def chat_response(content, finish="stop"):
    return {
        "id": "gen-test",
        "model": RETURNED_MODEL,
        "choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1200, "completion_tokens": 800, "cost": 0.0021},
    }


BRIDGE_SCRIPT = r"""
import json, sys
state_path, args = sys.argv[1], sys.argv[2:]
st = json.load(open(state_path))
st.setdefault("calls", []).append({"args": args, "stdin": sys.stdin.read() if args[0] != "pending" else ""})
json.dump(st, open(state_path, "w"), ensure_ascii=False)
if st.get("refuse", {}).get(args[0]):
    sys.stderr.write("refused: " + st["refuse"][args[0]])
    sys.exit(2)
if args[0] == "pending":
    print(json.dumps(st["rows"], ensure_ascii=False))
"""


@pytest.fixture()
def env(tmp_path, monkeypatch):
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


def ledger_row(env):
    db = sqlite3.connect(env["state"] / "ledger.sqlite")
    db.row_factory = sqlite3.Row
    r = db.execute("SELECT * FROM runner_runs WHERE agenda_id = ?", (AGENDA_ID,)).fetchone()
    revs = [dict(x) for x in db.execute("SELECT id, status FROM research_revisions")]
    db.close()
    return (dict(r) if r else None), revs


def bridge_calls(env, verb):
    return [c for c in env["bridge"]().get("calls", []) if c["args"][0] == verb]


def no_key_leak(env):
    for p in env["state"].rglob("*"):
        if p.is_file() and p.suffix != ".sqlite":
            text = p.read_text(encoding="utf-8", errors="replace")
            assert DRAFT_KEY not in text and JEV_KEY not in text, p


# ------------------------------------------------------------------------------------------------ pipeline
def test_happy_path_completes_and_reports_done(env):
    t = FakeOpenRouter([good_narrative()])
    assert env["run"](t) == 0
    rd = env["run_dir"]
    for f in ("config.json", "narrative.json", "revision.json", "run.json", "bars.csv"):
        assert (rd / f).is_file(), f
    run = json.loads((rd / "run.json").read_text())
    rev = json.loads((rd / "revision.json").read_text())
    assert run["status"] == "completed" and run["reasons"] == []
    assert run["models"] == {
        "draft_configured": draft.DEFAULT_MODEL,
        "draft_returned": [RETURNED_MODEL],
        "jev": runner.JEV_MODEL,
    }
    assert run["costs"]["draft_usd"] == 0.0021 and run["costs"]["jev_usd"] == pytest.approx(0.0003)
    assert run["draft_calls"][0]["prompt_sha256"] == draft.sha256_json(
        [{k: m[k] for k in ("role", "content")} for m in t.chat_calls[0]["messages"]]
    )
    assert {"assemble", "prepare", "draft1", "gates1", "jev", "complete"} <= set(run["timings_s"])
    # revision: same shape as the UMC reference (identity / research / manifest_payload / jev_results)
    assert set(rev) == {"identity", "research", "manifest_payload", "jev_results"}
    assert rev["identity"]["status"] == "completed" and rev["identity"]["replay_same_revision"] is True
    assert rev["jev_results"]["supports"] == rev["jev_results"]["n"] == 3
    assert rev["research"]["claims"][0]["quote"] == "Revenue was NT$68.73 billion for the quarter"
    ev = {e["id"]: e["kind"] for e in rev["manifest_payload"]["evidence"]}
    assert ev[DOC_ID] == "fundamental" and ev["kpi_series:2303"] == "kpi_series" and "ohlcv:0050" in ev
    cfg = json.loads((rd / "config.json").read_text())
    assert cfg["scope_key"] == "tw_stock:spot:2303" and cfg["symbols"] == ["2303", "0050"]
    assert cfg["model"] == draft.DEFAULT_MODEL and cfg["draft"]["reasoning"] is True
    # draft request: JSON mode, reasoning flag, generous max_tokens, model from config
    req = t.chat_calls[0]
    assert req["response_format"] == {"type": "json_object"} and req["reasoning"] == {"enabled": True}
    assert req["max_tokens"] == 16000 and req["model"] == draft.DEFAULT_MODEL
    assert "Revenue was NT$68.73 billion" in req["messages"][1]["content"]
    # bridge + ledger
    done = bridge_calls(env, "done")
    assert len(done) == 1 and done[0]["args"][1:] == [AGENDA_ID, rev["identity"]["revision_id"]]
    assert done[0]["stdin"].strip() == good_narrative()["tl_dr"]
    rec, revs = ledger_row(env)
    assert rec["status"] == "completed" and rec["reported"] == 1
    assert [r["status"] for r in revs] == ["completed"]
    assert all(h["Authorization"] in (f"Bearer {DRAFT_KEY}", f"Bearer {JEV_KEY}") for h in t.headers_seen)
    no_key_leak(env)


def test_precheck_refusal_fixed_by_repair_round(env):
    t = FakeOpenRouter([bad_number_narrative(), good_narrative()])
    assert env["run"](t) == 0
    run = json.loads((env["run_dir"] / "run.json").read_text())
    assert run["status"] == "completed"
    assert len(t.chat_calls) == 2
    assert any("[precheck]" in r and "claims[1]" in r for r in run["gates"][0]["reasons"])
    assert run["gates"][1]["reasons"] == []
    repair_msgs = t.chat_calls[1]["messages"]
    assert repair_msgs[-2]["role"] == "assistant" and "35.2%" in repair_msgs[-2]["content"]
    assert "validator refused" in repair_msgs[-1]["content"] and "claims[1]" in repair_msgs[-1]["content"]
    assert (env["run_dir"] / "narrative.draft1.json").is_file()
    assert len(bridge_calls(env, "done")) == 1


def test_persistent_refusal_needs_review_and_revision_stays_prepared(env):
    t = FakeOpenRouter([bad_number_narrative(), bad_number_narrative()])
    assert env["run"](t) == 0
    run = json.loads((env["run_dir"] / "run.json").read_text())
    rev = json.loads((env["run_dir"] / "revision.json").read_text())
    assert run["status"] == "needs_review" and run["failed_stage"] == "gates"
    assert len(t.chat_calls) == 2 and t.jev_calls == []  # one repair round, no escalation, no JEV
    assert rev["identity"]["status"] == "prepared" and rev["research"] is None and rev["needs_review"]
    rec, revs = ledger_row(env)
    assert rec["status"] == "needs_review" and [r["status"] for r in revs] == ["prepared"]
    nr = bridge_calls(env, "needs-review")
    assert len(nr) == 1 and nr[0]["args"][1:] == [AGENDA_ID] and bridge_calls(env, "done") == []
    assert nr[0]["stdin"].startswith("gates: validator refused") and len(nr[0]["stdin"]) <= 600
    assert "numeric mismatch" in nr[0]["stdin"]


def test_non_atomic_and_bad_quote_are_refused(env):
    n = good_narrative()
    n["claims"][0]["quote"] = "Revenue was NT$70 billion"  # not in the source
    n["claims"][1]["text"] = "聯電第三季毛利率為 32.5%，並且產能利用率為 85%。"
    t = FakeOpenRouter([n, n])
    env["run"](t)
    run = json.loads((env["run_dir"] / "run.json").read_text())
    reasons = run["gates"][0]["reasons"]
    assert any(r.startswith("[quote] claims[0]") for r in reasons)
    assert any(r.startswith("[atomic] claims[1]") for r in reasons)
    assert run["status"] == "needs_review"


def test_jev_non_supports_needs_review(env):
    t = FakeOpenRouter([good_narrative()], jev_label=lambda c: "says_nothing" if "價格事實" in c else "supports")
    assert env["run"](t) == 0
    run = json.loads((env["run_dir"] / "run.json").read_text())
    rev = json.loads((env["run_dir"] / "revision.json").read_text())
    assert run["status"] == "needs_review" and run["failed_stage"] == "jev"
    assert rev["jev_results"]["says_nothing"] == 1 and rev["identity"]["status"] == "prepared"
    assert len(t.jev_calls) == 3
    assert ledger_row(env)[1][0]["status"] == "prepared"
    assert "JEV did not support 1/3" in bridge_calls(env, "needs-review")[0]["stdin"]


@pytest.mark.parametrize(
    ("response", "needle"),
    [
        (chat_response(None), "content=None"),
        (chat_response('{"thesis": "cut', finish="length"), "finish_reason=length"),
        ((429, b'{"error": "rate limited"}'), "HTTP 429"),
    ],
)
def test_draft_failures_need_review(env, response, needle):
    t = FakeOpenRouter([response])
    assert env["run"](t) == 0
    run = json.loads((env["run_dir"] / "run.json").read_text())
    assert run["status"] == "needs_review" and run["failed_stage"] == "draft"
    assert needle in run["reasons"][0]
    assert len(t.chat_calls) == 1 and t.jev_calls == []
    assert needle in bridge_calls(env, "needs-review")[0]["stdin"]


def test_missing_or_unsafe_draft_key_is_a_named_error(env, capsys):
    t = FakeOpenRouter([good_narrative()])
    empty = env["tmp"] / "secrets" / "empty.env"
    empty.write_text(f"{draft.DRAFT_KEY_VAR}=\nOTHER_KEY=abc\n")
    empty.chmod(0o600)
    assert env["run"](t, draft_env_path=empty) == runner.EXIT_KEY
    assert f"{draft.DRAFT_KEY_VAR} missing or empty" in capsys.readouterr().err
    assert env["run"](t, draft_env_path=env["tmp"] / "nope.env") == runner.EXIT_KEY
    loose = env["tmp"] / "secrets" / "loose.env"
    loose.write_text(f"{draft.DRAFT_KEY_VAR}=x\n")
    loose.chmod(0o644)
    assert env["run"](t, draft_env_path=loose) == runner.EXIT_KEY
    assert "want 600" in capsys.readouterr().err
    assert t.chat_calls == [] and env["bridge"]().get("calls") is None  # fail closed before the bridge


def test_load_key_never_reads_process_env(tmp_path, monkeypatch):
    monkeypatch.setenv(draft.DRAFT_KEY_VAR, "from-env")
    f = tmp_path / "k.env"
    f.write_text("# comment\nexport OTHER=1\n")
    f.chmod(0o600)
    with pytest.raises(draft.MissingKeyError):
        draft.load_key(f, draft.DRAFT_KEY_VAR)
    f.write_text(f"{draft.DRAFT_KEY_VAR}='quoted-value'\n")
    assert draft.load_key(f, draft.DRAFT_KEY_VAR) == "quoted-value"


def test_idempotent_rerun_reports_again_without_new_work(env):
    t = FakeOpenRouter([good_narrative()])
    assert env["run"](t) == 0
    rev_id = ledger_row(env)[0]["revision_id"]
    # bridge still lists the row as accepted (e.g. the earlier done did not land): report done again, no new run
    assert env["run"](t) == 0
    assert len(t.chat_calls) == 1 and len(t.jev_calls) == 3
    done = bridge_calls(env, "done")
    assert len(done) == 2 and done[1]["args"][1:] == [AGENDA_ID, rev_id]
    # bridge no longer lists it: nothing to do, no bridge write
    env["set_bridge"](rows=[])
    assert env["run"](t) == 0
    assert len(bridge_calls(env, "done")) == 2


def test_bridge_pending_failure(env):
    env["set_bridge"](refuse={"pending": "key not allowed"})
    t = FakeOpenRouter([])
    assert env["run"](t) == runner.EXIT_BRIDGE_PENDING
    assert "bridge pending refused: refused: key not allowed" in (env["state"] / "runner.log").read_text()
    assert t.chat_calls == []


def test_bridge_done_refused_is_logged_and_retried(env):
    env["set_bridge"](refuse={"done": "unknown id"})
    t = FakeOpenRouter([good_narrative()])
    assert env["run"](t) == runner.EXIT_BRIDGE_REPORT
    rec, _ = ledger_row(env)
    assert rec["status"] == "completed" and rec["reported"] == 0
    assert "bridge done refused" in (env["state"] / "runner.log").read_text()
    env["set_bridge"](refuse={})
    assert env["run"](t) == 0
    assert ledger_row(env)[0]["reported"] == 1 and len(t.chat_calls) == 1


def test_assemble_source_failure_needs_review(env):
    row = dict(ROW, primary_source_url="https://ir.example.test/missing.htm")
    env["set_bridge"](rows=[row])
    t = FakeOpenRouter([])
    assert env["run"](t) == 0
    run = json.loads((env["run_dir"] / "run.json").read_text())
    assert run["status"] == "needs_review" and run["failed_stage"] == "assemble"
    assert "HTTP 404" in bridge_calls(env, "needs-review")[0]["stdin"]
    assert t.chat_calls == [] and not (env["run_dir"] / "revision.json").exists()


def test_assemble_falls_back_to_mirror(env):
    row = dict(ROW, primary_source_url="https://ir.example.test/missing.htm", mirror_url="https://m.test/umc-3q26.htm")
    env["set_bridge"](rows=[row])
    t = FakeOpenRouter([good_narrative()])
    assert env["run"](t) == 0
    cfg = json.loads((env["run_dir"] / "config.json").read_text())
    assert cfg["documents"][0]["source_uri"] == "https://m.test/umc-3q26.htm"
    assert any("mirror" in n for n in cfg["assemble_notes"])
    assert json.loads((env["run_dir"] / "run.json").read_text())["status"] == "completed"


def test_assemble_thalassa_failure_needs_review(env):
    def broken(symbol, start, end):
        raise ConnectionError("connection refused")

    kw = dict(env["assemble_kw"], fetch_bars=broken)
    t = FakeOpenRouter([])
    assert env["run"](t, kw=kw) == 0
    reason = bridge_calls(env, "needs-review")[0]["stdin"]
    assert reason.startswith("assemble: Thalassa OHLCV fetch failed for 2303")


def test_twse_failure_only_drops_kpi(env):
    def broken():
        raise TimeoutError("twse down")

    kw = dict(env["assemble_kw"], fetch_rows=broken)
    t = FakeOpenRouter([good_narrative()])
    assert env["run"](t, kw=kw) == 0
    cfg = json.loads((env["run_dir"] / "config.json").read_text())
    assert "kpi_symbols" not in cfg and any("TWSE" in n for n in cfg["assemble_notes"])
    assert json.loads((env["run_dir"] / "run.json").read_text())["status"] == "completed"


def test_dry_run_needs_no_keys_and_writes_nothing_to_the_bridge(env):
    t = FakeOpenRouter([])
    rows = env["tmp"] / "rows.json"
    rows.write_text(json.dumps([ROW]))
    argv = [
        "--state-dir",
        str(env["state"]),
        "--bridge",
        "/nonexistent-bridge",
        "--dry-run",
        "--agenda-json",
        str(rows),
    ]
    rc = runner.main(argv, transport=t, assemble_kw=env["assemble_kw"], as_of=AS_OF)
    assert rc == 0
    rd = env["state"] / "dry-run" / "runs" / AGENDA_ID
    prompt = json.loads((rd / "prompt.json").read_text())
    assert prompt[0]["role"] == "system" and "zh-TW" in prompt[0]["content"]
    assert DOC_ID in prompt[1]["content"] and "kpi_series:2303" in prompt[1]["content"]
    assert json.loads((rd / "run.json").read_text())["status"] == "dry_run"
    assert t.chat_calls == [] and not (env["state"] / "ledger.sqlite").exists()


def test_lock_busy_exits_quietly(env):
    import fcntl

    env["state"].mkdir(parents=True)
    with open(env["state"] / "runner.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        t = FakeOpenRouter([good_narrative()])
        assert env["run"](t) == 0
    assert t.chat_calls == [] and env["bridge"]().get("calls") is None


def test_malformed_ids_never_reach_the_bridge(env):
    b = runner.Bridge(f"{sys.executable} -c pass")
    with pytest.raises(runner.BridgeError):
        b.done("2026-10-21", "not-a-uuid", "x")
    with pytest.raises(runner.BridgeError):
        b.needs_review("20261021-3; rm -rf /", "x")


def test_template_and_scope_mapping():
    assert [assemble.template_for(t) for t in ("earnings", "investor_conference", "press_release", "other")] == [
        "earnings",
        "earnings",
        "press_release",
        "generic",
    ]
    assert assemble.scope_for(["2303"]) == ("tw_stock:spot:2303", "2303")
    assert assemble.scope_for(["2408", "2344"]) == ("tw_stock:basket:2344-2408", None)


def test_excerpt_cap(tmp_path):
    paras = "\n\n".join(
        f"<p>Revenue line {i} grew strongly this quarter according to the company quarterly report text body.</p>"
        for i in range(60)
    )
    f = tmp_path / "many.htm"
    f.write_text(f"<html><body>{paras}</body></html>")
    doc, stats = assemble.fetch_document(
        f.as_uri(),
        doc_id="fundamental:x",
        title="t",
        publisher="p",
        event_time="2026-10-01T00:00:00Z",
        out_dir=str(tmp_path / "out"),
        template="earnings",
    )
    assert stats["verified"] == 60 and len(doc["excerpts"]) == assemble.MAX_EXCERPTS


def test_draft_schema_is_the_narrative_schema_plus_quote():
    """research_theme narrative fields (as in the UMC v2 narrative, minus variant_perception which needs consensus)
    plus a per-claim verbatim `quote`."""
    assert set(draft.NARRATIVE_SCHEMA) == {
        "tl_dr",
        "thesis",
        "stance",
        "change_summary",
        "next_review_at",
        "catalysts",
        "risks",
        "invalidation_conditions",
        "unknowns",
        "claims",
    }
    assert set(draft.NARRATIVE_SCHEMA["claims"][0]) == {"kind", "text", "evidence_ids", "quote"}


def test_outbox_mode_queues_reports_and_host_delivery(env):
    import deliver_outbox

    outbox = env["tmp"] / "outbox"
    t = FakeOpenRouter([good_narrative()])
    assert env["run"](t, "--outbox", str(outbox)) == 0
    assert bridge_calls(env, "done") == [] and len(bridge_calls(env, "pending")) == 1
    entry = json.loads((outbox / f"{AGENDA_ID}.json").read_text())
    rev_id = ledger_row(env)[0]["revision_id"]
    assert entry == {"verb": "done", "args": [AGENDA_ID, rev_id], "stdin": good_narrative()["tl_dr"] + "\n"}
    bridge_cmd = env["bridge_cmd"]
    env["set_bridge"](refuse={"done": "already done"})
    assert deliver_outbox.main(["--outbox", str(outbox), "--bridge", bridge_cmd]) == 4
    assert (outbox / "refused" / f"{AGENDA_ID}.json").is_file()
    (outbox / "refused" / f"{AGENDA_ID}.json").replace(outbox / f"{AGENDA_ID}.json")
    env["set_bridge"](refuse={})
    assert deliver_outbox.main(["--outbox", str(outbox), "--bridge", bridge_cmd]) == 0
    assert not (outbox / f"{AGENDA_ID}.json").exists()
    assert bridge_calls(env, "done")[-1]["args"][1:] == [AGENDA_ID, rev_id]
    (outbox / "20261021-9.json").write_text(json.dumps({"verb": "done", "args": ["20261021-9", "x; rm"], "stdin": ""}))
    assert deliver_outbox.main(["--outbox", str(outbox), "--bridge", bridge_cmd]) == 4
    assert len(bridge_calls(env, "done")) == 2  # malformed entry never reaches the bridge
