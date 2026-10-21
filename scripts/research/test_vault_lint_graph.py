"""Tests for the Nexus vault lint (_kb/ops/lint/lint.py) and graph export (_kb/ops/compile/graph_export.py).

The scripts live in the vault; they are run as subprocesses against a temporary fake vault via NEXUS_VAULT,
so the real vault is only read for the script source. stdlib + pyyaml only.
"""

import datetime as dt
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REAL_VAULT = Path(
    os.environ.get("NEXUS_VAULT_SCRIPTS")
    or os.path.expanduser("~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Nexus")
)
LINT = REAL_VAULT / "_kb" / "ops" / "lint" / "lint.py"
GRAPH = REAL_VAULT / "_kb" / "ops" / "compile" / "graph_export.py"
TODAY = dt.date.today()
OLD = (TODAY - dt.timedelta(days=60)).isoformat()
PAST = (TODAY - dt.timedelta(days=3)).isoformat()
FUTURE = (TODAY + dt.timedelta(days=7)).isoformat()

pytestmark = pytest.mark.skipif(not LINT.is_file() or not GRAPH.is_file(), reason="vault scripts not present")

TAXONOMY = """\
---
type: reference
title: Taxonomy
status: draft
created: 2026-10-21
updated: 2026-10-21
tags: [taxonomy, schema]
origin: agent
sources: ["[[_kb/raw/src]]"]
---

```yaml
sectors:
  foundry: {title: "foundry"}
  memory: {title: "memory"}
```

```yaml
concepts:
  capacity-utilization: {title: "cu"}
  hbm: {title: "hbm"}
```
"""


def note(fm, body=""):
    return "---\n" + textwrap.dedent(fm).strip() + "\n---\n\n" + body + "\n"


def base(type_, title, status="active", created="2026-10-01", tags="[]", origin="agent", extra=""):
    return 'type: %s\ntitle: "%s"\nstatus: %s\ncreated: %s\nupdated: %s\ntags: %s\norigin: %s\n%s' % (
        type_,
        title,
        status,
        created,
        created,
        tags,
        origin,
        extra,
    )


def research(rev, scope, as_of, stance, extra="", tags='[poseidon, tw_stock, "2303", research]'):
    return note(
        base(
            "research",
            rev,
            tags=tags,
            extra='scope_key: "%s"\nrevision_id: "%s"\nas_of: %sT04:00:00Z\n'
            "stance: %s\n%s" % (scope, rev, as_of, stance, extra),
        )
    )


def write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.fixture()
def vault(tmp_path):
    v = tmp_path / "Nexus"
    write(v, "AGENTS.md", "# fake vault\n")
    write(v, "_kb/wiki/_taxonomy.md", TAXONOMY)
    write(v, "_kb/raw/src.md", note(base("source", "src", extra='compiled: true\ncompiled_into: ["[[foundry]]"]')))
    src = 'sources: ["[[_kb/raw/src]]"]\n'
    write(v, "_kb/wiki/sectors/foundry.md", note(base("sector", "Foundry", extra=src)))
    write(v, "_kb/wiki/sectors/memory.md", note(base("sector", "Memory", extra=src)))  # no inbound -> orphan
    write(v, "_kb/wiki/concepts/capacity-utilization.md", note(base("concept", "CU", extra=src)))
    write(v, "_kb/wiki/concepts/hbm.md", note(base("concept", "HBM", status="stub", created=OLD)))  # stale stub
    write(
        v,
        "_kb/wiki/companies/2303-umc.md",
        note(
            base(
                "company",
                "UMC",
                tags='[company, "2303"]',
                extra='symbol: "2303"\nsector: "[[sectors/foundry]]"\naliases: ["UMC"]\n',
            ),
            "See [[UMC]] alias.",
        ),
    )
    write(v, "_kb/wiki/companies/2330-tsmc.md", note(base("company", "TSMC", extra='symbol: "2330"\n')))
    write(
        v,
        "_kb/wiki/companies/9999-bad.md",
        note(base("company", "Bad", extra='symbol: "9999"\nsector: "[[sectors/bogus]]"\n')),
    )
    write(v, "_kb/wiki/timelines/2026-2303.md", note(base("timeline", "2026 2303")))
    # 2303 chain: aaa (old, superseded by bbb) -> bbb (latest, review passed, bullish)
    write(
        v,
        "Research/tw_stock-spot-2303/aaa.md",
        research("aaa", "tw_stock:spot:2303", "2026-09-01", "watch", 'next_review_at: "%s"\n' % PAST),
    )
    write(
        v,
        "Research/tw_stock-spot-2303/bbb.md",
        research(
            "bbb",
            "tw_stock:spot:2303",
            "2026-10-01",
            "bullish",
            'next_review_at: "%sT00:00:00+08:00"\nprevious_revision_id: "aaa"\ncompany: "[[companies/2303-umc]]"\n'
            'sector: "[[sectors/foundry]]"\nconcepts: ["[[concepts/capacity-utilization]]"]\n'
            'peers: ["[[companies/2330-tsmc]]"]\ntimeline: "[[timelines/2026-2303]]"\n'
            'related: ["[[aaa]]", "[[Research/tw_stock-spot-2303/aaa]]"]\n' % PAST,
        )
        + "Body [[companies/2303-umc|UMC]] [[nowhere-page]] `[[in-code]]` [[2303-umc\\|UMC]]\n",
    )
    # another scope for the same symbol with the opposite stance and bad taxonomy ids
    write(
        v,
        "Research/tw_stock-event-2303/ccc.md",
        research(
            "ccc",
            "tw_stock:event:2303",
            "2026-10-02",
            "bearish",
            'company: "[[companies/2303-umc]]"\nconcepts: ["[[concepts/made-up]]"]\nnext_review_at: "%s"\n' % FUTURE,
            tags='[poseidon, tw_stock, decision-loop, "2303", micron, concept/foo, Capacity_Utilization, my-new-idea]',
        ),
    )
    write(
        v,
        "Research/tw_stock-spot-2330/ddd.md",
        research("ddd", "tw_stock:spot:2330", "2026-10-01", "watch", tags='[poseidon, "2330"]'),
    )  # orphan, about by symbol
    write(
        v,
        "Research/tw_stock-sector-memory/eee.md",
        research("eee", "tw_stock:sector:memory", "2026-10-01", "watch", 'sector: "[[sectors/memory]]"\n'),
    )
    # things the walk must skip or never read
    write(v, ".obsidian/junk.md", "no frontmatter [[dead-in-obsidian]]\n")
    write(v, "Research/tw_stock-spot-2303/.tmp-aaa.123", "partial")
    write(v, "Nexus-Work/secret.md", "no frontmatter [[dead-in-work]]\n")
    write(v, "Soak/2026-10-20.md", '---\ndate: "2026-10-20"\nstatus: "red"\n---\n\n# Soak\n')
    return v


def run(script, v, *args):
    env = dict(os.environ, NEXUS_VAULT=str(v))
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, env=env)


def findings(out):
    """{rule: [line, ...]} parsed from the printed report."""
    rules, cur = {}, None
    for line in out.splitlines():
        if line.startswith("## "):
            cur = line.split("`")[1]
            rules[cur] = []
        elif cur and line.startswith("- "):
            rules[cur].append(line)
    return rules


def has(rules, rule, *needles):
    return any(all(n in line for n in needles) for line in rules.get(rule, []))


def test_lint_rules_on_fake_vault(vault):
    r = run(LINT, vault)
    assert r.returncode == 1, r.stdout + r.stderr  # taxonomy ERRORs present
    f = findings(r.stdout)
    # walk sees Research/ but skips .obsidian, .tmp*, the -Work vault and soak schema
    assert "dead-in-obsidian" not in r.stdout and "dead-in-work" not in r.stdout and ".tmp-aaa" not in r.stdout
    assert "Nexus-Work" not in r.stdout and "Soak/" not in r.stdout
    assert has(f, "research-orphan", "Research/tw_stock-spot-2330/ddd.md")
    assert not has(f, "research-orphan", "bbb.md") and not has(f, "research-orphan", "eee.md")
    assert has(f, "company-no-sector", "_kb/wiki/companies/2330-tsmc.md")
    assert has(f, "sector-orphan", "_kb/wiki/sectors/memory.md") is False  # eee links memory via `sector`
    assert has(f, "concept-orphan", "_kb/wiki/concepts/hbm.md")
    assert not has(f, "concept-orphan", "capacity-utilization")
    assert not has(f, "sector-orphan", "foundry")
    assert has(f, "stale-thesis", "Research/tw_stock-spot-2303/bbb.md", PAST)
    assert not has(f, "stale-thesis", "aaa.md") and not has(f, "stale-thesis", "ccc.md")
    assert has(f, "conflicting-stance", "Research/tw_stock-event-2303/ccc.md", "tw_stock-spot-2303/bbb.md")
    assert has(f, "taxonomy-id", "9999-bad.md", "sectors/bogus")
    assert has(f, "taxonomy-id", "ccc.md", "concepts/made-up")
    assert has(f, "taxonomy-id", "ccc.md", "concept/foo")
    assert has(f, "taxonomy-id", "ccc.md", "Capacity_Utilization")
    assert has(f, "taxonomy-id", "ccc.md", "my-new-idea")
    assert not has(f, "taxonomy-id", "micron") and not has(f, "taxonomy-id", "`2303`")
    assert not has(f, "taxonomy-id", "`tw_stock`") and not has(f, "taxonomy-id", "`decision-loop`")
    assert has(f, "stale-stub", "_kb/wiki/concepts/hbm.md")
    assert has(f, "dead-link", "bbb.md", "nowhere-page")
    for ok in ("companies/2303-umc", "Research/tw_stock-spot-2303/aaa", "[[aaa]]", "[[UMC]]", "in-code", "2303-umc\\"):
        assert not has(f, "dead-link", ok), ok
    assert has(f, "research-agent-origin", "bbb.md")
    assert "research-origin" not in f


def test_lint_write_and_clean_exit(vault, tmp_path):
    r = run(LINT, vault, "--write")
    out = vault / "_kb" / "logs" / "lint-reports" / ("%s-lint.md" % TODAY.isoformat())
    assert out.is_file() and r.stdout.startswith("ERROR ")
    text = out.read_text(encoding="utf-8")
    assert "| ERROR | taxonomy-id |" in text and "## WARN `stale-thesis`" in text
    # a vault with only warnings exits 0, and the dry run writes nothing
    clean = tmp_path / "clean"
    write(clean, "AGENTS.md", "x\n")
    write(clean, "_kb/wiki/_taxonomy.md", TAXONOMY)
    write(
        clean,
        "Research/tw_stock-spot-2330/ddd.md",
        research("ddd", "tw_stock:spot:2330", "2026-10-01", "watch", tags="[poseidon]"),
    )
    r2 = run(LINT, clean)
    assert r2.returncode == 0, r2.stdout
    assert "research-orphan" in r2.stdout
    assert not (clean / "_kb" / "logs").exists()


def test_lint_refuses_non_vault(tmp_path):
    r = run(LINT, tmp_path)
    assert r.returncode != 0 and "refusing" in r.stderr


def test_graph_export(vault):
    r = run(GRAPH, vault)
    assert r.returncode == 0, r.stderr
    snap = vault / "_kb" / "graph" / "snapshots" / ("%s.json" % TODAY.isoformat())
    rep = vault / "_kb" / "graph" / "reports" / ("%s-graph.md" % TODAY.isoformat())
    first = snap.read_bytes()
    g = json.loads(first)
    nodes = {n["id"]: n for n in g["nodes"]}
    edges = {(e["from"], e["to"], e["kind"]) for e in g["edges"]}
    assert nodes["_kb/wiki/companies/2303-umc"]["type"] == "company"
    assert nodes["Research/tw_stock-spot-2303/bbb"]["extra"] == {
        "stance": "bullish",
        "as_of": "2026-10-01T04:00:00",
        "scope_key": "tw_stock:spot:2303",
    }
    assert nodes["Research/tw_stock-spot-2303/bbb"]["path"] == "Research/tw_stock-spot-2303/bbb.md"
    assert not any(i.startswith(("Nexus-Work", ".obsidian", "Soak/", "_kb/graph")) for i in nodes)
    b = "Research/tw_stock-spot-2303/bbb"
    for e in [
        (b, "_kb/wiki/companies/2303-umc", "about"),
        (b, "_kb/wiki/sectors/foundry", "in_sector"),
        (b, "_kb/wiki/companies/2330-tsmc", "read_across"),
        (b, "_kb/wiki/concepts/capacity-utilization", "uses_concept"),
        (b, "Research/tw_stock-spot-2303/aaa", "predecessor"),
        (b, "_kb/wiki/timelines/2026-2303", "catalyst"),
        ("_kb/wiki/companies/2303-umc", "_kb/wiki/sectors/foundry", "in_sector"),
        ("Research/tw_stock-spot-2330/ddd", "_kb/wiki/companies/2330-tsmc", "about"),
    ]:
        assert e in edges, e
    # typed pairs are not duplicated as generic wikilinks (bbb links aaa and 2303-umc in related/body too)
    assert (b, "Research/tw_stock-spot-2303/aaa", "wikilink") not in edges
    assert (b, "_kb/wiki/companies/2303-umc", "wikilink") not in edges
    assert {e["kind"] for e in g["edges"]} <= {
        "about",
        "in_sector",
        "read_across",
        "uses_concept",
        "predecessor",
        "supersedes",
        "catalyst",
        "wikilink",
    }
    text = rep.read_text(encoding="utf-8")
    assert "## Top-10 hubs by degree" in text and "| company |" in text and "orphans" in text
    assert "sectors/bogus" in text  # unresolved typed edge reported
    r2 = run(GRAPH, vault)
    assert r2.returncode == 0 and snap.read_bytes() == first and rep.read_text(encoding="utf-8") == text
