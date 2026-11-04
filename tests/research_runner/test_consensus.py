"""Offline tests for research_theme.py consensus and surprise evidence (gap B1) and variant perception (B5).

SQLite ledger, synthetic bars, no network (the Yahoo probe gets a fake opener). Split from test_offline_tooling.py.
  PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/research_runner/test_consensus.py
"""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from testkit import ACTUAL_Q4, EVENT_IDX, N, day, doc_with, go, render


@pytest.fixture()
def env(theme_env):
    return theme_env


# ------------------------------------------------------------------------------------------------ consensus / surprise (B1), variant perception (B5)
# SYNTHETIC consensus numbers (test fixtures only; not real analyst estimates).
def consensus_file(env, as_of_day=15, items=None, sym="AAA", name="cons", tamper=False):
    raw = env["tmp"] / f"{name}-raw.html"
    raw.write_text("<html>synthetic consensus capture for tests</html>")
    data = {
        "symbol": sym,
        "provider": "synthetic-test",
        "as_of": f"{day(as_of_day)}T00:00:00Z",
        "source_uri": "https://example.com/consensus/AAA",
        "raw_path": str(raw),
        "raw_sha256": "0" * 64 if tamper else hashlib.sha256(raw.read_bytes()).hexdigest(),
        "fetched_at": f"{day(as_of_day)}T00:00:00Z",
        "items": items
        or [
            {
                "metric": "revenue",
                "period": "Q4 2026",
                "mean": 52e9,
                "low": 51e9,
                "high": 53e9,
                "n_analysts": 20,
                "unit": "USD",
                "currency": "USD",
                "basis": "IFRS",
            },
            {
                "metric": "gross_margin",
                "period": "4Q26",
                "mean": 61.5,
                "median": 61.4,
                "low": 60.5,
                "high": 62.0,
                "n_analysts": 18,
                "unit": "%",
                "basis": "IFRS",
            },
            {
                "metric": "eps",
                "period": "FY2026",
                "mean": 10.0,
                "low": 9.0,
                "high": 11.0,
                "n_analysts": 12,
                "unit": "USD",
                "currency": "USD",
                "basis": "IFRS",
            },
        ],
    }
    path = env["tmp"] / f"{name}.json"
    path.write_text(json.dumps(data))
    return ["--consensus-json", str(path)]


def surprise_case(env, name="sur", cons_day=15, narrative_extra=None, cfg_extra=None, items=None):
    cfg = copy.deepcopy(env["cfg"])
    cfg["documents"] = [doc_with(env, name, ACTUAL_Q4)]
    cfg["actuals"] = [
        {
            "metric": "revenue",
            "period": "4Q26",
            "value": 53.1,
            "unit": "USD billion",
            "evidence_id": "fundamental:doc1",
            "excerpt_locator": "line 2",
        }
    ]
    cfg.update(cfg_extra or {})
    n = copy.deepcopy(env["narrative"])
    n["claims"].append(
        {
            "kind": "fact",
            "text": "Q4 gross margin was 61.2%.",
            "evidence_ids": ["fundamental:doc1"],
            "actual": {"metric": "gross margin", "period": "Q4'26", "value": 61.2, "unit": "%"},
        }
    )
    n.update(narrative_extra or {})
    return go(env, name, cfg, narrative=n, extra=consensus_file(env, cons_day, items=items, name=name + "-c"))


def test_consensus_frozen_and_surprise_rendered(env):
    p, d = surprise_case(env)
    assert p.returncode == 0, p.stderr
    rev = json.loads((d / "rev.json").read_text())
    ev = {e["id"]: e for e in rev["manifest_payload"]["evidence"]}
    c = ev["consensus:AAA"]
    assert (
        c["kind"] == "consensus"
        and c["payload"]["source_tier"] == "third-party aggregate"
        and c["payload"]["items"][0]["period"] == "4Q26"
    )
    assert (
        c["payload"]["items"][2]["period"] == "FY26"
        and len(c["payload"]["raw_sha256"]) == 64
        and c["event_time"] == f"{day(15)}T00:00:00Z"
    )
    rows = {r["metric"]: r for r in ev["surprise:AAA"]["payload"]["rows"]}
    assert ev["surprise:AAA"]["payload"]["algorithm"] == "surprise-v1" and set(rows) == {"revenue", "gross_margin"}
    rv, gm = rows["revenue"], rows["gross_margin"]
    assert (
        rv["status"] == "ok"
        and rv["surprise_abs"] == 1.1e9
        and rv["surprise_pct"] == round(1.1 / 52 * 100, 2)
        and rv["vs_range"] == "above"
    )
    assert (
        gm["status"] == "ok"
        and gm["surprise_abs"] == -0.3
        and gm["surprise_pct"] == round(-0.3 / 61.5 * 100, 2)
        and gm["vs_range"] == "within"
    )
    assert gm["actual"]["source"] == "claims[3]" and gm["actual"]["published_at"] == f"{day(EVENT_IDX)}T10:00:00Z"
    from poseidon.research_runner.consensus import replay_surprise

    assert replay_surprise(rev["manifest_payload"]) == {"surprise:AAA": True}
    tampered = copy.deepcopy(rev["manifest_payload"])
    next(e for e in tampered["evidence"] if e["id"] == "surprise:AAA")["payload"]["rows"][0]["surprise_abs"] = 0
    assert replay_surprise(tampered) == {"surprise:AAA": False}
    md = render(d / "rev.json").stdout
    assert "## Consensus & surprise" in md
    assert (
        "| revenue | 4Q26 | 52.00 billion USD（51.00 billion USD–53.00 billion USD，n=20） | 53.1 USD billion（fundamental:doc1） | +1.10 billion USD（+2.12%） | above |"
        in md
    ), md
    assert "| gross_margin | 4Q26 | 61.5%（60.5%–62%，n=18） | 61.2 %（claims[3]） | -0.3 pp（-0.49%） | within |" in md
    assert "| eps | FY26 | 10 USD（9 USD–11 USD，n=12） | n/a | n/a | n/a |" in md
    tail = md[md.index("## 揭露（Disclosure）") :]
    assert "`consensus:AAA`：third-party aggregate" in tail and "`surprise:AAA`：derived" in tail


def test_consensus_changes_manifest_hash_and_claim_precheck(env):
    p0, _ = go(env, "c0")
    p1, _ = go(env, "c1", extra=consensus_file(env, name="c1c"))
    assert p0.returncode == p1.returncode == 0, p1.stderr
    assert json.loads(p0.stdout)["manifest_sha256"] != json.loads(p1.stdout)["manifest_sha256"]
    n = copy.deepcopy(env["narrative"])
    n["claims"].append(
        {"kind": "fact", "text": "Analysts expected 4Q26 revenue of US$52 billion.", "evidence_ids": ["consensus:AAA"]}
    )
    p, d = go(env, "c2", narrative=n, extra=consensus_file(env, name="c2c"))
    assert p.returncode == 0, p.stderr
    assert json.loads((d / "rev.json").read_text())["identity"]["claim_precheck"][3]["numbers"] == "ok"
    n["claims"][3]["text"] = "Analysts expected 4Q26 revenue of US$5.2 billion."
    p, d = go(env, "c3", narrative=n, extra=consensus_file(env, name="c3c"))
    assert p.returncode != 0 and "claims[3] numeric mismatch" in p.stderr


def test_post_event_consensus_gives_no_surprise_number(env):
    p, d = surprise_case(env, "post", cons_day=25)  # observed after the actual was published (day 19)
    assert p.returncode == 0, p.stderr
    rows = next(
        e for e in json.loads((d / "rev.json").read_text())["manifest_payload"]["evidence"] if e["kind"] == "surprise"
    )["payload"]["rows"]
    assert all(r["status"].startswith("post-event") and r["surprise_abs"] is None for r in rows)
    assert "post-event (consensus observed" in render(d / "rev.json").stdout


def test_unit_mismatch_and_late_snapshot(env):
    items = [
        {
            "metric": "revenue",
            "period": "4Q26",
            "mean": 1600.0,
            "low": 1500.0,
            "high": 1700.0,
            "n_analysts": 5,
            "unit": "TWD billion",
            "currency": "TWD",
            "basis": "IFRS",
        }
    ]
    p, d = surprise_case(env, "fx", items=items)
    assert p.returncode == 0, p.stderr
    row = next(
        e for e in json.loads((d / "rev.json").read_text())["manifest_payload"]["evidence"] if e["kind"] == "surprise"
    )["payload"]["rows"][0]
    assert row["status"].startswith("n/a (unit differs") and row["surprise_abs"] is None  # never FX-converts
    p, _ = go(
        env, "late", extra=consensus_file(env, as_of_day=N + 3, name="latec")
    )  # observed after the manifest as_of
    assert p.returncode == 0 and "not point-in-time, skipped" in p.stderr and "consensus:AAA" not in p.stdout


@pytest.mark.parametrize(
    "mutate,msg",
    [
        (lambda d: d.update(raw_sha256="0" * 64), "refusing to freeze"),
        (lambda d: d["items"][0].update(low=53e9, high=54e9), "low <= mean <= high"),
        (lambda d: d["items"][0].update(period="next quarter"), "does not normalise"),
        (lambda d: d["items"][0].update(n_analysts=0), "n_analysts"),
        (lambda d: d["items"][0].update(currency="TWD"), "does not match unit"),
        (lambda d: d.pop("source_uri"), "source_uri must be a non-empty string"),
    ],
)
def test_consensus_json_refusals(env, mutate, msg):
    args = consensus_file(env, name="bad")
    path = Path(args[1])
    data = json.loads(path.read_text())
    mutate(data)
    path.write_text(json.dumps(data))
    p, _ = go(env, "cbad", extra=args)
    assert p.returncode != 0 and msg in p.stderr, p.stderr


VP = {
    "metric": "revenue",
    "period": "4Q26",
    "consensus_ref": "consensus:AAA",
    "our_view": 54e9,
    "direction": "above",
    "why": "backlog in the doc",
}


def test_variant_perception_and_pilot_position_rendered(env):
    vp = [
        VP,
        dict(
            VP, metric="gross_margin", our_view="margins hold despite mix", direction="in_line", why="no new fab costs"
        ),
    ]
    p, d = surprise_case(
        env,
        "vp",
        narrative_extra={"variant_perception": vp},
        cfg_extra={"pilot_position": {"position": "long", "size_note": "1 lot paper"}},
    )
    assert p.returncode == 0, p.stderr
    rev = json.loads((d / "rev.json").read_text())
    assert (
        rev["research"]["variant_perception"][0]["direction"] == "above"
        and rev["research"]["disclosure"]["pilot_position"] == "long"
    )
    md = render(d / "rev.json").stdout
    assert "## Variant perception（推論；我們 vs 共識）" in md
    assert (
        "| revenue | 4Q26 | 52.00 billion USD（`consensus:AAA`） | 54.00 billion USD | above | backlog in the doc |"
        in md
    )
    assert (
        "| gross_margin | 4Q26 | 61.5%（`consensus:AAA`） | margins hold despite mix | in_line | no new fab costs |"
        in md
    )
    assert (
        "- 部位揭露：pilot_position long（1 lot paper），as_of " in md
        and "未記錄" not in md.split("部位揭露")[1].split("\n")[0]
    )


@pytest.mark.parametrize(
    "vp,msg",
    [
        (dict(VP, our_view=50e9), "inconsistent with direction 'above'"),
        (dict(VP, direction="in_line", our_view=55e9), "inconsistent with direction 'in_line'"),
        (dict(VP, why=""), "variant_perception[0].why"),
        (dict(VP, consensus_ref="fundamental:doc1"), "is not a frozen consensus evidence"),
        (dict(VP, period="1Q27"), "has no revenue 1Q27 item"),
        (dict(VP, direction="higher"), "direction must be one of"),
    ],
)
def test_variant_perception_refusals(env, vp, msg):
    p, d = surprise_case(env, "vpbad", narrative_extra={"variant_perception": [vp]})
    assert p.returncode != 0 and msg in p.stderr, p.stderr
    assert not (d / "rev.json").exists()


def test_pilot_position_invalid_and_undisclosed(env):
    cfg = copy.deepcopy(env["cfg"])
    cfg["pilot_position"] = "maybe"
    p, _ = go(env, "ppbad", cfg, narrative=env["narrative"])
    assert p.returncode != 0 and "pilot_position must be one of" in p.stderr
    cfg["pilot_position"] = "undisclosed"
    p, d = go(env, "ppund", cfg, narrative=env["narrative"])
    assert p.returncode == 0, p.stderr
    assert "- 部位揭露：pilot_position undisclosed，as_of" in render(d / "rev.json").stdout


class FakeResp:
    def __init__(self, status, body):
        self.status, self.body = status, body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeOpener:
    """Maps URL prefixes to (status, body); records every request. No network."""

    def __init__(self, routes):
        self.routes, self.seen = routes, []

    def open(self, req, timeout=None):
        import io
        import urllib.error

        url = req.full_url
        self.seen.append(url)
        st, body = next((v for k, v in self.routes.items() if url.startswith(k)), (404, b"not found"))
        if st != 200:
            raise urllib.error.HTTPError(url, st, "x", {}, io.BytesIO(body))
        return FakeResp(st, body)


def yahoo_page(trend):
    body = json.dumps({"quoteSummary": {"result": [{"earningsTrend": {"trend": trend}}], "error": None}})
    blob = json.dumps({"status": 200, "body": body}).replace("<", "\\u003c")
    return (
        '<html><script type="application/json" data-sveltekit-fetched data-url="https://query1.finance.yahoo.com/v10/finance/quoteSummary/TSM?'
        'formatted=true&amp;modules=earningsTrend%2CcalendarEvents">' + blob + "</script></html>"
    ).encode()


TREND = [
    {
        "period": "0q",
        "endDate": "2026-12-31",
        "earningsEstimate": {
            "avg": {"raw": 5.16},
            "low": {"raw": 5.1},
            "high": {"raw": 5.28},
            "numberOfAnalysts": {"raw": 8},
            "earningsCurrency": "USD",
        },
        "revenueEstimate": {
            "avg": {"raw": 1.705e12},
            "low": {"raw": 1.626e12},
            "high": {"raw": 1.753e12},
            "numberOfAnalysts": {"raw": 26},
            "revenueCurrency": "TWD",
        },
    },
    {
        "period": "0y",
        "endDate": "2026-12-31",
        "earningsEstimate": {
            "avg": 17.5,
            "low": 17.18,
            "high": 17.95,
            "numberOfAnalysts": 11,
            "earningsCurrency": "USD",
        },
        "revenueEstimate": {},
    },
    {"period": "+5y", "endDate": None, "growth": {"raw": 0.2}},
]


def test_yahoo_probe_parses_embedded_quote_summary(tmp_path):
    from poseidon.research_runner.consensus import fetch_yahoo_consensus

    op = FakeOpener({"https://finance.yahoo.com/quote/TSM/analysis/": (200, yahoo_page(TREND))})
    data, st = fetch_yahoo_consensus("2330", "TSM", tmp_path / "raw", adr=True, opener=op)
    assert st["status"] == "ok" and st["attempts"][0]["status"] == 200 and len(st["attempts"][0]["sha256"]) == 64
    assert [(i["metric"], i["period"], i["unit"], i["n_analysts"]) for i in data["items"]] == [
        ("eps_per_adr", "4Q26", "USD", 8),
        ("revenue", "4Q26", "TWD", 26),
        ("eps_per_adr", "FY26", "USD", 11),
    ]
    assert hashlib.sha256(Path(data["raw_path"]).read_bytes()).hexdigest() == data["raw_sha256"] and data[
        "source_uri"
    ].endswith("/TSM/analysis/")


def test_yahoo_probe_failures_are_clean(tmp_path):
    from poseidon.research_runner.consensus import fetch_yahoo_consensus

    cf = FakeOpener({"https://finance.yahoo.com/": (200, b"<html><title>Just a moment...</title>cf-chl</html>")})
    data, st = fetch_yahoo_consensus("2303", "UMC", tmp_path, opener=cf)
    assert data is None and st["status"] == "unavailable" and "Cloudflare" in st["reason"] and len(cf.seen) == 1
    limited = FakeOpener(
        {
            "https://finance.yahoo.com/": (200, b"<html>no embedded data</html>"),
            "https://fc.yahoo.com/": (404, b""),
            "https://query2.finance.yahoo.com/v1/test/getcrumb": (429, b"Too Many Requests"),
        }
    )
    data, st = fetch_yahoo_consensus("2303", "UMC", tmp_path, opener=limited)
    assert (
        data is None and "getcrumb HTTP 429" in st["reason"] and all("crumb=" not in a["url"] for a in st["attempts"])
    )

    class Down:
        def open(self, req, timeout=None):
            raise OSError("network unreachable")

    data, st = fetch_yahoo_consensus("2303", "UMC", tmp_path, opener=Down())
    assert data is None and st["status"] == "unavailable"


def test_consensus_live_without_mapping_does_not_fail(env):
    cfg = copy.deepcopy(env["cfg"])
    cfg["consensus_live"] = True
    cfg["adr_ticker"] = {}  # no mapping -> no network call
    p, d = go(env, "live", cfg, narrative=env["narrative"])
    assert p.returncode == 0, p.stderr
    rev = json.loads((d / "rev.json").read_text())
    assert rev["identity"]["consensus_status"] == [
        {"symbol": "AAA", "status": "unavailable", "reason": "no adr_ticker mapping for a TW symbol"}
    ]
    assert "共識不可用：AAA：unavailable（no adr_ticker mapping for a TW symbol）" in render(d / "rev.json").stdout


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("3Q26", "3Q26"),
        ("Q3 2026", "3Q26"),
        ("2026Q3", "3Q26"),
        ("3Q2026", "3Q26"),
        ("Q3'26", "3Q26"),
        ("FY2026", "FY26"),
        ("fy26", "FY26"),
        ("1H26", "1H26"),
        ("Q4", "Q4"),
    ],
)
def test_norm_period(raw, expected):
    from poseidon.research_runner.consensus import norm_period

    assert norm_period(raw) == expected
