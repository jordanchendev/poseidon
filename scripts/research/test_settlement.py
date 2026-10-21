"""Offline tests for settle_theme.py (settle-v1). SQLite ledger built by research_theme.main itself, synthetic bars.

PYTHONPATH=src python -m pytest -q -p no:cacheprovider scripts/research/test_settlement.py  (from the Poseidon repo root)
"""

import json
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import research_theme  # noqa: E402
import settle_theme  # noqa: E402

START = date(2026, 9, 1)
N, DECIDE, HORIZON = 40, 25, 30  # decided as_of = day 25 09:00Z; horizon = day 30 01:00Z; bars at 00:00Z
SETTLE_AS_OF = "2026-10-15T00:00:00Z"  # = day 44 > every bar; future horizon case uses day 50


def day(i):
    return (START + timedelta(days=i)).isoformat()


# close paths: BBB is the benchmark (+0.2%/day); AAA strongly outperforms after the decision, CCC falls, DDD tracks BBB.
def bbb(i):
    return 100.0 * (1.002**i)


def aaa(i):
    return bbb(i) * (1.0 if i <= DECIDE else 1.0 + 0.01 * (i - DECIDE))


def ccc(i):
    return bbb(i) * (1.0 if i <= DECIDE else 1.0 - 0.01 * (i - DECIDE))


def ddd(i):
    return bbb(i) * (1.0 if i <= DECIDE else 1.0 + 0.001 * (i - DECIDE))


PATHS = {"AAA": aaa, "BBB": bbb, "CCC": ccc, "DDD": ddd}


def rel_pct(f):
    """Expected settle-v1 rel return: (f(h)/f(d) - bbb(h)/bbb(d)) * 100 on unrounded fractions."""
    return round(((f(HORIZON) / f(DECIDE) - 1) - (bbb(HORIZON) / bbb(DECIDE) - 1)) * 100, 2)


def write_csv(path, paths=PATHS, n=N):
    rows = [
        f"{day(i)}T00:00:00Z,{s},{f(i)},{f(i) * 1.01},{f(i) * 0.99},{f(i)},{1000 + i}"
        for s, f in paths.items()
        for i in range(n)
    ]
    path.write_text("\n".join(rows) + "\n")


def make_revision(tmp, ledger, csv, name, symbols, stance, horizon_day=HORIZON, symbol=None, scope=None, extra=None):
    """One completed revision through research_theme.main (real ManifestService/ResearchService schema)."""
    subj = [s for s in symbols if s != "BBB"]
    cfg = {
        "scope_key": scope or f"tw_stock:spot:{subj[0]}",
        "as_of": f"{day(DECIDE)}T09:00:00Z",
        "event_time": f"{day(DECIDE - 5)}T10:00:00Z",
        "account_scope": "paper:test",
        "universe_id": "u-test",
        "thalassa_base_url": "http://127.0.0.1:1",
        "symbols": symbols,
        "benchmark": "BBB",
        "required_data": {"ohlcv": {"max_age_seconds": 172800}},
        "policy_version": f"test/{name}",
        "provider": "test",
        "model": "test",
        "documents": [],
    }
    if symbol:
        cfg["symbol"] = symbol
    s0 = subj[0]
    narrative = {
        "thesis": f"Thesis {name}.",
        "stance": stance,
        "change_summary": "First version.",
        "next_review_at": f"{day(horizon_day)}T01:00:00Z review after the horizon bar",
        "risks": ["r"],
        "invalidation_conditions": ["i"],
        "unknowns": ["u"],
        "claims": [
            {"kind": "fact", "text": f"{s0} closed above its 20-day average.", "evidence_ids": [f"price_facts:{s0}"]},
            {"kind": "fact", "text": f"{s0} traded every day in the window.", "evidence_ids": [f"ohlcv:{s0}"]},
            {"kind": "inference", "text": f"{s0} may keep its trend.", "evidence_ids": [f"price_facts:{s0}"]},
        ],
    }
    narrative.update(extra or {})
    d = tmp / name
    d.mkdir()
    (d / "cfg.json").write_text(json.dumps(cfg))
    (d / "narr.json").write_text(json.dumps(narrative))
    argv = sys.argv
    sys.argv = [
        "research_theme.py",
        "--config",
        str(d / "cfg.json"),
        "--csv",
        str(csv),
        "--ledger",
        str(ledger),
        "--narrative",
        str(d / "narr.json"),
        "--out",
        str(d / "rev.json"),
    ]
    try:
        research_theme.main()
    finally:
        sys.argv = argv
    return json.loads((d / "rev.json").read_text())["identity"]["revision_id"].replace(
        "-", ""
    )  # ledger stores uuid hex


def settle(ledger, csv, *extra, as_of=SETTLE_AS_OF):
    settle_theme.main(["settle", "--ledger", str(ledger), "--csv", str(csv), "--as-of", as_of, *extra])


def rows(ledger):
    db = sqlite3.connect(ledger)
    db.row_factory = sqlite3.Row
    return {r["revision_id"]: dict(r) for r in db.execute("SELECT * FROM research_settlements")}


@pytest.fixture()
def world(tmp_path):
    csv = tmp_path / "bars.csv"
    write_csv(csv)
    ledger = tmp_path / "ledger.sqlite"
    ids = {
        "long_aaa": make_revision(tmp_path, ledger, csv, "long_aaa", ["AAA", "BBB"], "long", symbol="AAA"),
        "long_ccc": make_revision(tmp_path, ledger, csv, "long_ccc", ["CCC", "BBB"], "long", symbol="CCC"),
        "long_ddd": make_revision(tmp_path, ledger, csv, "long_ddd", ["DDD", "BBB"], "long", symbol="DDD"),
        "watch_ddd": make_revision(
            tmp_path, ledger, csv, "watch_ddd", ["DDD", "BBB"], "watch", symbol="DDD", scope="tw_stock:theme:ddd"
        ),
        "watch_aaa": make_revision(
            tmp_path, ledger, csv, "watch_aaa", ["AAA", "BBB"], "watch", symbol="AAA", scope="tw_stock:theme:aaa"
        ),
        "future": make_revision(
            tmp_path,
            ledger,
            csv,
            "future",
            ["CCC", "BBB"],
            "short",
            horizon_day=50,
            symbol="CCC",
            scope="tw_stock:theme:ccc",
        ),
        "basket": make_revision(
            tmp_path, ledger, csv, "basket", ["AAA", "CCC", "BBB"], "watch", scope="tw_stock:sector:mix"
        ),
    }
    return dict(tmp=tmp_path, csv=csv, ledger=ledger, ids=ids)


def test_outcomes_hit_miss_flat_unsettled(world, capsys):
    settle(world["ledger"], world["csv"])
    r = rows(world["ledger"])
    i = world["ids"]
    assert (
        rel_pct(aaa) > 1 and rel_pct(ccc) < -1 and abs(rel_pct(ddd)) <= 1
    )  # the synthetic paths do what the names say
    assert r[i["long_aaa"]]["outcome"] == "hit" and r[i["long_aaa"]]["rel_return_pct"] == rel_pct(aaa)
    assert r[i["long_aaa"]]["abs_return_pct"] == round((aaa(HORIZON) / aaa(DECIDE) - 1) * 100, 2)
    assert r[i["long_ccc"]]["outcome"] == "miss" and r[i["long_ccc"]]["rel_return_pct"] == rel_pct(ccc)
    assert r[i["long_ddd"]]["outcome"] == "flat"
    assert (
        r[i["watch_ddd"]]["outcome"] == "hit" and r[i["watch_aaa"]]["outcome"] == "miss"
    )  # neutral: |rel|<=1 hit, else miss
    assert i["future"] not in r  # horizon day 50 > settle as_of: not settled, not written
    b = r[i["basket"]]
    assert b["symbol"] == "AAA,CCC" and "equal-weight basket" in b["detail"]
    assert b["rel_return_pct"] == round(
        ((aaa(HORIZON) / aaa(DECIDE) + ccc(HORIZON) / ccc(DECIDE)) / 2 - bbb(HORIZON) / bbb(DECIDE)) * 100, 2
    )
    row = r[i["long_aaa"]]
    assert (
        row["settle_algorithm"] == "settle-v1"
        and row["benchmark"] == "BBB"
        and row["horizon_time"] == f"{day(HORIZON)}T01:00:00Z"
    )
    assert row["horizon_bar_time"] == f"{day(HORIZON)}T00:00:00Z" and len(row["bars_sha256"]) == 64
    assert row["content_sha256"] == settle_theme.row_hash(row)
    settle_theme.main(["pending", "--ledger", str(world["ledger"]), "--as-of", SETTLE_AS_OF])
    out = capsys.readouterr().out
    assert "## awaiting horizon (1)" in out and i["future"] in out


def test_matches_price_facts_v2_math(world):
    """settle-v1 rel == research_theme.stats() rel_ret_since_event_pct on the same window (shared _event_returns)."""
    bars = research_theme.load_bars(world["csv"])
    cut = settle_theme.parse_ts(f"{day(HORIZON)}T01:00:00Z")
    facts = research_theme.stats(bars["AAA"].loc[:cut], f"{day(DECIDE)}T09:00:00Z", bars["BBB"].loc[:cut])
    settle(world["ledger"], world["csv"])
    assert rows(world["ledger"])[world["ids"]["long_aaa"]]["rel_return_pct"] == facts["rel_ret_since_event_pct"]


def test_idempotent_and_force(world, capsys):
    settle(world["ledger"], world["csv"])
    first = rows(world["ledger"])
    capsys.readouterr()
    settle(world["ledger"], world["csv"])
    assert rows(world["ledger"]) == first
    assert capsys.readouterr().out.count("no-op: already") == 6
    write_csv(world["csv"], {**PATHS, "DDD": ccc})  # bars changed after the fact (e.g. a corrected feed)
    settle(world["ledger"], world["csv"])
    assert rows(world["ledger"]) == first  # still a no-op without --force
    settle(world["ledger"], world["csv"], "--force")
    after = rows(world["ledger"])
    rid = world["ids"]["long_ddd"]
    assert after[rid]["outcome"] == "miss" and after[rid]["bars_sha256"] != first[rid]["bars_sha256"]
    assert after[rid]["content_sha256"] != first[rid]["content_sha256"]


def test_flag_before_and_after_settlement_survives_force(world, capsys):
    i = world["ids"]
    L = str(world["ledger"])
    settle_theme.main(
        ["flag", "--ledger", L, "--revision", i["long_aaa"], "--note", "SMA5 crossed above SMA20 on day 27"]
    )
    r = rows(world["ledger"])[i["long_aaa"]]
    assert r["invalidation_flag"] == 1 and r["outcome"] == "unsettled" and r["note"].startswith("SMA5")
    settle(world["ledger"], world["csv"])
    r = rows(world["ledger"])[i["long_aaa"]]
    assert (
        r["outcome"] == "hit" and r["invalidation_flag"] == 1 and r["note"].startswith("SMA5")
    )  # unsettled rows are retried
    settle(world["ledger"], world["csv"], "--force")
    r = rows(world["ledger"])[i["long_aaa"]]
    assert r["invalidation_flag"] == 1 and r["content_sha256"] == settle_theme.row_hash(r)
    settle_theme.main(["flag", "--ledger", L, "--revision", i["long_aaa"], "--note", "false alarm", "--clear"])
    assert rows(world["ledger"])[i["long_aaa"]]["invalidation_flag"] == 0
    with pytest.raises(SystemExit, match="not a completed revision"):
        settle_theme.main(["flag", "--ledger", L, "--revision", "deadbeef", "--note", "x"])


def test_report(world, capsys):
    settle(world["ledger"], world["csv"])
    capsys.readouterr()
    settle_theme.main(
        ["flag", "--ledger", str(world["ledger"]), "--revision", world["ids"]["watch_aaa"], "--note", "thesis broke"]
    )
    out_path = world["tmp"] / "report.md"
    settle_theme.main(["report", "--ledger", str(world["ledger"]), "--as-of", SETTLE_AS_OF, "--out", str(out_path)])
    md = out_path.read_text()
    assert "| long | 3 | 1 | 1 | 1 | 0 | 0 |" in md
    assert (
        "| watch | 3 | 2 | 1 | 0 | 0 | 1 |" in md
    )  # watch_ddd hit, basket hit (AAA +x, CCC -x cancel), watch_aaa miss
    assert "| short | 1 | 0 | 0 | 0 | 1 | 0 | n/a |" in md
    assert "awaiting horizon" in md and "flag: thesis broke" in md


def test_unsettled_when_csv_short_or_unknown_stance(world, capsys):
    short = world["tmp"] / "short.csv"
    write_csv(short, n=HORIZON - 2)  # csv stops before the horizon
    settle(world["ledger"], short)
    r = rows(world["ledger"])
    assert (
        all(v["outcome"] == "unsettled" for v in r.values())
        and "does not reach the horizon" in r[world["ids"]["long_aaa"]]["detail"]
    )
    settle(world["ledger"], world["csv"])
    assert rows(world["ledger"])[world["ids"]["long_aaa"]]["outcome"] == "hit"  # retried
    assert settle_theme.direction_of("Bullish") == 1 and settle_theme.direction_of("maybe") is None


def test_adjusted_csv_base_mismatch_refused(world):
    adj = world["tmp"] / "adj.csv"
    write_csv(adj, {**PATHS, "AAA": lambda i: aaa(i) * 0.98})  # e.g. dividend-adjusted series
    settle(world["ledger"], adj)
    assert "!= frozen manifest close" in rows(world["ledger"])[world["ids"]["long_aaa"]]["detail"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2026-10-21T06:40:00Z（下一個時槽）", "2026-10-21T06:40:00Z"),
        ("2026-10-21T06:40:00+08:00 x", "2026-10-20T22:40:00Z"),
        ("2026-11-10 monthly revenue", "2026-11-10T23:59:59Z"),
        ("after the call", None),
    ],
)
def test_parse_horizon(text, expected):
    assert settle_theme.iso(settle_theme.parse_horizon(text)) == expected


# ---------------------------------------------------------------- superseded (C3)
def dashed(h):
    return "%s-%s-%s-%s-%s" % (h[:8], h[8:12], h[12:16], h[16:20], h[20:])


@pytest.fixture()
def chain(tmp_path):
    csv = tmp_path / "bars.csv"
    write_csv(csv)
    ledger = tmp_path / "ledger.sqlite"
    ids = {
        "v1": make_revision(tmp_path, ledger, csv, "v1", ["AAA", "BBB"], "long", symbol="AAA"),
        "early": make_revision(
            tmp_path,
            ledger,
            csv,
            "early",
            ["CCC", "BBB"],
            "short",
            horizon_day=50,
            symbol="CCC",
            scope="tw_stock:theme:ccc",
        ),
    }
    return dict(tmp=tmp_path, csv=csv, ledger=ledger, ids=ids)


def test_superseded_excluded_and_converted_without_force(chain, capsys):
    t, i = chain, chain["ids"]
    settle(t["ledger"], t["csv"])
    assert rows(t["ledger"])[i["v1"]]["outcome"] == "hit"
    # later revision on the same scope corrects v1 (dashed id, as printed in revision JSON); `supersedes` key form for the other
    i["v2"] = make_revision(
        t["tmp"],
        t["ledger"],
        t["csv"],
        "v2",
        ["AAA", "BBB"],
        "long",
        symbol="AAA",
        extra={"supersedes_revision_id": dashed(i["v1"])},
    )
    i["early2"] = make_revision(
        t["tmp"],
        t["ledger"],
        t["csv"],
        "early2",
        ["CCC", "BBB"],
        "short",
        horizon_day=50,
        symbol="CCC",
        scope="tw_stock:theme:ccc",
        extra={"supersedes": i["early"]},
    )
    capsys.readouterr()
    settle(t["ledger"], t["csv"])
    out = capsys.readouterr().out
    r = rows(t["ledger"])
    v1 = r[i["v1"]]
    assert v1["outcome"] == "superseded" and v1["outcome_before_supersede"] == "hit" and v1["superseded_by"] == i["v2"]
    assert (
        v1["rel_return_pct"] == rel_pct(aaa)
        and v1["content_sha256"] == settle_theme.row_hash(v1)
        and "superseded by" in out
    )
    e = r[i["early"]]  # superseded before its horizon: never measured
    assert (
        e["outcome"] == "superseded"
        and e["outcome_before_supersede"] is None
        and "superseded before its horizon" in e["detail"]
    )
    assert r[i["v2"]]["outcome"] == "hit" and r[i["v2"]]["superseded_by"] is None and i["early2"] not in r
    settle(t["ledger"], t["csv"])
    assert rows(t["ledger"]) == r  # idempotent
    md_path = t["tmp"] / "r.md"
    settle_theme.main(["report", "--ledger", str(t["ledger"]), "--as-of", SETTLE_AS_OF, "--out", str(md_path)])
    md = md_path.read_text()
    assert "| long | 2 | 1 | 0 | 0 | 0 | 0 | %+.2f | 1 | 1/1 |" % rel_pct(aaa) in md  # v1 counted only as superseded
    assert "| short | 2 | 0 | 0 | 0 | 1 | 0 | n/a | 1 | n/a |" in md
    assert "superseded (was hit)" in md and "2 superseded" in md


def test_supersede_pointer_must_exist(chain):
    with pytest.raises(SystemExit, match="not a completed revision in this ledger"):
        make_revision(
            chain["tmp"],
            chain["ledger"],
            chain["csv"],
            "bad",
            ["AAA", "BBB"],
            "long",
            symbol="AAA",
            extra={"supersedes_revision_id": "00000000-0000-0000-0000-000000000000"},
        )


def test_supersede_map_ignores_backward_pointer():
    t = lambda n: "2026-10-21 04:00:%02d" % n
    revs = [
        {"id": "a", "supersedes": "b", "created_at": t(1)},
        {"id": "b", "supersedes": None, "created_at": t(2)},
        {"id": "c", "supersedes": "b", "created_at": t(2)},
        {"id": "d", "supersedes": "b", "created_at": t(3)},
        {"id": "e", "supersedes": "zzz", "created_at": t(4)},
    ]
    # a was created before b (backward pointer, ignored); c ties with b (same second, allowed); first successor wins; unknown ignored
    assert settle_theme.supersede_map(revs) == {"b": "c"}


def test_pre_supersede_table_migrates_with_hashes_intact(world):
    settle(world["ledger"], world["csv"])
    before = rows(world["ledger"])
    db = sqlite3.connect(world["ledger"])
    old_cols = [c for c in settle_theme.COLUMNS if c not in settle_theme.SUPERSEDE_COLUMNS]
    db.execute("ALTER TABLE research_settlements RENAME TO tmp_s")
    db.execute(
        settle_theme.DDL.replace(",'superseded'", "")
        .split("outcome_before_supersede TEXT")[0]
        .rstrip()
        .rstrip(",")
        .replace("flagged_at TEXT", "flagged_at TEXT, content_sha256 TEXT NOT NULL")
        + ")"
    )
    db.execute("INSERT INTO research_settlements SELECT %s FROM tmp_s" % ",".join(old_cols))
    db.execute("DROP TABLE tmp_s")
    db.commit()
    assert "superseded_by" not in [r[1] for r in db.execute("PRAGMA table_info(research_settlements)")]
    settle_theme.connect(world["ledger"]).close()
    after = rows(world["ledger"])
    assert after == before and all(r["content_sha256"] == settle_theme.row_hash(r) for r in after.values())
