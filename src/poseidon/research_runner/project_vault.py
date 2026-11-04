"""Project a completed offline research revision (revision JSON from research_theme.py) into the Nexus vault.

Writes Research/<scope_key path-safe>/<revision_id>.md atomically, never overwrites, frontmatter per the
2026-10-15 vault schema plus the research fields; predecessor goes to `previous_revision_id` and `related`.
Path-safe scope: ':' -> '-'. origin stays 'agent' until Poseidon owns the projection (v21).

Knowledge-graph edges (design 2026-10-21-vault-knowledge-graph-linking.md §3.2-3.3), added to the frontmatter and
repeated as bullets in a trailing `## 知識連結` section:
  company   "[[companies/<symbol>-<slug>]]" (spot scope: the scope symbol; sector scope: every ohlcv symbol except
            --benchmark). Slug comes from the existing company page whose `symbol` matches, else --company-slug,
            else the bare symbol. One target -> string, several -> list.
  sector    the company pages' `sector` (+ `sectors/<id>` for a sector scope listed in the taxonomy). String/list as above.
  concepts  list; taxonomy concept keywords matched against thesis + claims + risks + invalidation_conditions.
            Only ids from _kb/wiki/_taxonomy.md (YAML blocks) are ever emitted.
  peers     list; read_across symbols from the revision identity (revisions made before 2026-10-29 only).
  themes    list, only when a claim carries theme_ids (design 2026-10-29 §3.4): "[[themes/<theme_id>]]" in claim
            order. The ids passed the runner's closed-list gate; _kb/wiki/themes/ is theme_pages.py's, never stubbed
            here, and lint's theme-id rule checks them against _kb/wiki/themes/_registry.md.
  (timeline edges and _kb/wiki/timelines/ pages were retired 2026-10-21; catalysts live in the company block.
   `timeline` stays in EDGE_KEYS only so --replace strips it from older notes.)
plus the machine fields research_links.py reads: next_review_at (leading ISO timestamp of the narrative's
next_review_at), summary (tl_dr, else first sentence of the thesis), catalysts (list), event_status
(pre_event|post_event, only when the revision's narrative has it; set by the research runner).
Missing link targets get a stub page (status: stub, origin: agent); existing pages are never overwritten.

--replace: allowed only when the existing note has the same revision_id, the same content_sha256 and origin: agent.
The note is rewritten in place (temp file + rename): every existing frontmatter line and the body are kept byte for
byte; only the edge/machine fields and the `## 知識連結` section are (re)generated. The body is NOT re-rendered
(render_theme.py has changed since the older notes were projected).

After writing, runs <vault>/_kb/ops/compile/research_links.py --apply (if present) so the company
blocks are regenerated, and appends `## [date] research | <symbol> <rev8> <stance>` to _kb/wiki/log.md.
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

VAULT = os.environ.get("NEXUS_VAULT") or os.path.expanduser(
    "~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Nexus"
)
EDGE_KEYS = (
    "company",
    "sector",
    "concepts",
    "peers",
    "themes",
    "timeline",
    "next_review_at",
    "summary",
    "catalysts",
    "event_status",
)
LINKS_HEAD = "## 知識連結"
LINK = re.compile(r"\[\[([^\]|#]+)")
LOG_HEADER = """---
type: index
title: Wiki Log
status: active
created: {today}
updated: {today}
tags: [log]
origin: agent
---

# Wiki Log

Append-only operations log (Karpathy LLM-wiki convention). One line per event, **newest at the bottom**:
`## [YYYY-MM-DD] <kind> | <detail>` with kind in `research` (project_vault.py), `compile` (compile.py),
`research-links` (research_links.py), `ingest`, `lint`. Never edit, delete or reorder past lines.
Read with `grep '^## \\[' _kb/wiki/log.md | tail`.

"""


def today():
    return datetime.date.today().isoformat()


def append_log(vault, line):
    p = vault / "_kb" / "wiki" / "log.md"
    if (p.parent / ".log.md.icloud").exists():
        sys.exit("refusing: %s is an iCloud placeholder; download it first" % p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(LOG_HEADER.format(today=today()), encoding="utf-8")
    with open(p, "a", encoding="utf-8") as f:
        f.write("## [%s] %s\n" % (today(), line))


def split(text):
    """-> (frontmatter text without the --- fences, body after the closing fence line) or (None, text)."""
    if text.startswith("---\n"):
        end = text.find("\n---\n", 3)
        if end != -1:
            return text[4:end], text[end + 5 :]
    return None, text


def load_fm(path):
    fm, _ = split(path.read_text(encoding="utf-8"))
    return (yaml.safe_load(fm) or {}) if fm is not None else {}


def load_taxonomy(vault):
    p = vault / "_kb" / "wiki" / "_taxonomy.md"
    if not p.exists():
        print("warning: %s missing; no concepts will be linked" % p, file=sys.stderr)
        return {}, {}
    sectors, concepts = {}, {}
    for block in re.findall(r"```yaml\n(.*?)```", p.read_text(encoding="utf-8"), re.S):
        data = yaml.safe_load(block) or {}
        sectors.update(data.get("sectors") or {})
        concepts.update(data.get("concepts") or {})
    return sectors, concepts


def keyword_hit(kw, text):
    if kw.isascii():
        return re.search(r"(?<![A-Za-z0-9])%s(?![A-Za-z0-9])" % re.escape(kw), text, re.I) is not None
    return kw in text


def match_concepts(concepts, r):
    text = "\n".join(
        [r.get("thesis") or ""]
        + [c.get("text", "") for c in r.get("claims") or []]
        + list(r.get("risks") or [])
        + list(r.get("invalidation_conditions") or [])
    )
    return [
        cid
        for cid, spec in concepts.items()
        if any(keyword_hit(str(k), text) for k in (spec or {}).get("keywords") or [])
    ]


def company_index(vault):
    out = {}
    for p in sorted((vault / "_kb" / "wiki" / "companies").glob("*.md")):
        sym = load_fm(p).get("symbol")
        if sym is not None:
            out[str(sym).lower()] = p.stem
    return out


def link(target):
    return "[[%s]]" % target


def one_or_list(targets):
    return (
        json.dumps(link(targets[0]), ensure_ascii=False)
        if len(targets) == 1
        else "[%s]" % ", ".join(json.dumps(link(t), ensure_ascii=False) for t in targets)
    )


def first_sentence(text):
    m = re.match(r"(.+?[。！？]|.+?\.(?=\s|$))", (text or "").strip(), re.S)
    return (m.group(1) if m else (text or "").strip()).replace("\n", " ")


def compute_edges(vault, i, r, scope_key, a):
    sectors_tax, concepts_tax = load_taxonomy(vault)
    cidx = company_index(vault)
    slugs = dict(s.split("=", 1) for s in a.company_slug)

    def resolve(sym):
        s = str(sym).lower()
        stem = cidx.get(s) or ("%s-%s" % (s, slugs[sym]) if sym in slugs else s)
        return "companies/" + stem

    parts = scope_key.split(":")
    if len(parts) == 3 and parts[1] == "sector":
        symbols = []
        for e in i["evidence"]:
            if e["id"].startswith("ohlcv:"):
                s = e["id"].split(":", 1)[1]
                if s != a.benchmark and s not in symbols:
                    symbols.append(s)
    else:
        symbols = [str(r.get("symbol") or parts[-1])]
    companies = [resolve(s) for s in symbols]
    sectors = []
    for c in companies:
        p = vault / "_kb" / "wiki" / (c + ".md")
        m = LINK.search(str(load_fm(p).get("sector") or "")) if p.exists() else None
        if m and m.group(1).strip() not in sectors:
            sectors.append(m.group(1).strip())
    if len(parts) == 3 and parts[1] == "sector" and parts[2] in sectors_tax and "sectors/" + parts[2] not in sectors:
        sectors.append("sectors/" + parts[2])
    peers = []
    for x in i.get("read_across") or []:
        t = resolve(str(x["symbol"]))
        if t not in companies and t not in peers:
            peers.append(t)
    concepts = ["concepts/" + c for c in match_concepts(concepts_tax, r)]
    themes = ["themes/" + t for t in dict.fromkeys(t for c in r.get("claims") or [] for t in c.get("theme_ids") or [])]
    nr = re.match(r"\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2})?Z?)?", str(r.get("next_review_at") or ""))
    return dict(
        symbols=symbols,
        companies=companies,
        sectors=sectors,
        concepts=concepts,
        peers=peers,
        themes=themes,
        next_review_at=nr.group(0) if nr else None,
        summary=r.get("tl_dr") or first_sentence(r.get("thesis")),
        catalysts=[{k: c.get(k) for k in ("date", "type", "expected_signal")} for c in r.get("catalysts") or []],
        event_status=r.get("event_status") if r.get("event_status") in ("pre_event", "post_event") else None,
        sectors_tax=sectors_tax,
        concepts_tax=concepts_tax,
    )


def edge_lines(e):
    out = []
    if e["companies"]:
        out.append("company: " + one_or_list(e["companies"]))
    if e["sectors"]:
        out.append("sector: " + one_or_list(e["sectors"]))
    out.append("concepts: [%s]" % ", ".join(json.dumps(link(c), ensure_ascii=False) for c in e["concepts"]))
    out.append("peers: [%s]" % ", ".join(json.dumps(link(c), ensure_ascii=False) for c in e["peers"]))
    if e["themes"]:
        out.append("themes: [%s]" % ", ".join(json.dumps(link(t), ensure_ascii=False) for t in e["themes"]))
    if e["next_review_at"]:
        out.append("next_review_at: %s" % e["next_review_at"])
    if e["summary"]:
        out.append("summary: " + json.dumps(e["summary"], ensure_ascii=False))
    if e["catalysts"]:
        out.append("catalysts: [%s]" % ", ".join(json.dumps(c, ensure_ascii=False) for c in e["catalysts"]))
    if e["event_status"]:
        out.append("event_status: %s" % e["event_status"])
    return out


def links_section(e):
    rows = [
        ("公司", e["companies"]),
        ("板塊", e["sectors"]),
        ("概念", e["concepts"]),
        ("同業 / read-across", e["peers"]),
        ("主題", e["themes"]),
    ]
    bullets = ["- %s：%s" % (label, link(t)) for label, ts in rows for t in ts]
    return "\n%s\n\n%s\n" % (LINKS_HEAD, "\n".join(bullets) or "- （無）")


def strip_links(body):
    idx = body.rfind("\n%s\n" % LINKS_HEAD)
    return body[:idx] if idx != -1 else body


def write_new(path, text):
    """Create a page only if it does not exist (never overwrite). Returns True when created."""
    if path.exists() or (path.parent / (".%s.icloud" % path.name)).exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "x", encoding="utf-8") as f:
            f.write(text)
    except FileExistsError:
        return False
    return True


def stub(fm_lines, title, line, extra=""):
    return "---\n%s\n---\n\n# %s\n\n%s\n%s" % ("\n".join(fm_lines), title, line, extra)


def create_stubs(vault, e, rev8):
    created, d = [], today()
    wiki = vault / "_kb" / "wiki"
    base = lambda typ, title, tags: [
        "type: %s" % typ,
        "title: %s" % json.dumps(title, ensure_ascii=False),
        "status: stub",
        "created: %s" % d,
        "updated: %s" % d,
        "tags: [%s]" % ", ".join(json.dumps(t, ensure_ascii=False) for t in tags),
        "origin: agent",
    ]
    note = "Stub（%s，由 project_vault.py 投影 revision %s 時建立，待人補）。" % (d, rev8)
    for c in e["companies"] + e["peers"]:
        stem = c.split("/", 1)[1]
        sym = stem.split("-", 1)[0]
        title = "%s（%s）" % (stem.split("-", 1)[1], sym) if "-" in stem else sym
        block = (
            "\n## 研究時間線\n\n<!-- research-timeline:start (generated by _kb/ops/compile/research_links.py; do not edit) -->\n"
            "_尚未生成。_\n<!-- research-timeline:end -->\n"
        )
        fm = base("company", title, ["company", sym]) + [
            'symbol: "%s"' % sym,
            "sources: []",
            "related: []",
            "confidence: low",
        ]
        if write_new(wiki / (c + ".md"), stub(fm, title, note, block)):
            created.append(c)
    for typ, targets, tax in (
        ("sector", e["sectors"], e["sectors_tax"]),
        ("concept", e["concepts"], e["concepts_tax"]),
    ):
        for t in targets:
            cid = t.split("/", 1)[1]
            title = (tax.get(cid) or {}).get("title") or cid
            fm = base(typ, title, [typ, cid]) + ["sources: []", 'related: ["[[_taxonomy]]"]', "confidence: low"]
            if write_new(wiki / (t + ".md"), stub(fm, title, note)):
                created.append(t)
    return created


def evidence_ref(d):
    """Evidence dir as 'aquarium/<path>' when it lives under the aquarium root, so notes never carry a home path.
    A remote reference `host:path` (e.g. the stormtrooper runner's run dir, from nexus_pull.py) is kept as given."""
    if re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*:", str(d)):
        return str(d)
    p = Path(d).expanduser().resolve()
    # aquarium root = nearest ancestor of this script holding local_dev/ (the script lives in poseidon/src/poseidon/research_runner)
    root = next((q for q in Path(__file__).resolve().parents if (q / "local_dev").is_dir()), None)
    if root is None:
        return d
    try:
        return "aquarium/%s/" % p.relative_to(root).as_posix()
    except ValueError:
        return d


def atomic_write(path, text, rev8):
    tmp = path.parent / (".tmp-%s.%d" % (rev8, os.getpid()))
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("revision_json")
    ap.add_argument("--title", required=True)
    ap.add_argument("--evidence-dir", required=True)
    ap.add_argument("--note", default="", help="extra callout text (sources, blocks, topic origin)")
    ap.add_argument("--topic-source", default=None)
    ap.add_argument("--poseidon-commit", default="483c363bda5ec399dc22b02499f60cb3cd192057")
    ap.add_argument("--related", action="append", default=[])
    ap.add_argument("--out-dir", default=None, help="override vault dir (for tests)")
    ap.add_argument("--benchmark", default="0050", help="excluded from sector-scope company links")
    ap.add_argument(
        "--company-slug", action="append", default=[], metavar="SYMBOL=SLUG", help="slug for a company with no page yet"
    )
    ap.add_argument(
        "--replace",
        action="store_true",
        help="re-link an existing note (same revision_id + content_sha256, origin agent)",
    )
    a = ap.parse_args()
    d = json.load(open(a.revision_json))
    i = d["identity"]
    r = d["research"]
    if i.get("status") != "completed":
        sys.exit("revision is not completed")
    vault = Path(a.out_dir) if a.out_dir else Path(VAULT)
    if not a.out_dir and not (vault / "AGENTS.md").exists():
        sys.exit("vault not found at %s" % vault)
    scope_key = i.get("scope_key") or r["scope_key"]
    rev = i["revision_id"]
    rev8 = rev[:8]
    scope_dir = vault / "Research" / scope_key.replace(":", "-")
    scope_dir.mkdir(parents=True, exist_ok=True)
    target = scope_dir / ("%s.md" % rev)
    if (scope_dir / (".%s.icloud" % target.name)).exists():
        sys.exit("refusing: %s is an iCloud placeholder; download it first" % target)
    e = compute_edges(vault, i, r, scope_key, a)
    if target.exists():
        if not a.replace:
            sys.exit("refusing to overwrite %s" % target)
        fm_text, body = split(target.read_text(encoding="utf-8"))
        old = yaml.safe_load(fm_text or "") or {}
        if str(old.get("revision_id")) != rev:
            sys.exit("refusing --replace: revision_id %s != %s" % (old.get("revision_id"), rev))
        if str(old.get("content_sha256")) != i["content_sha256"]:
            sys.exit(
                "refusing --replace: content_sha256 mismatch (note %s, revision %s)"
                % (str(old.get("content_sha256"))[:12], i["content_sha256"][:12])
            )
        if old.get("origin") != "agent":
            sys.exit("refusing --replace: origin is %r, only origin: agent notes may be re-linked" % old.get("origin"))
        if old.get("title") != a.title:
            print("note: --title differs from the note's title; existing frontmatter is kept", file=sys.stderr)
        keep = [l for l in fm_text.split("\n") if not re.match(r"(%s):" % "|".join(EDGE_KEYS), l)]
        text = "---\n%s\n---\n%s%s" % ("\n".join(keep + edge_lines(e)), strip_links(body), links_section(e))
        atomic_write(target, text, rev8)
        action = "re-linked"
    else:
        rendered = subprocess.run(
            [sys.executable, "-m", "poseidon.research_runner.render_theme", a.revision_json],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        related = list(a.related)
        if i.get("previous_revision_id"):
            related.insert(0, i["previous_revision_id"])
        symbols = sorted({x["id"].split(":", 1)[1] for x in i["evidence"] if x["id"].startswith("ohlcv:")})
        fm = [
            "---",
            "type: research",
            'title: "%s"' % a.title.replace('"', "'"),
            "status: active",
            "created: %s" % today(),
            "updated: %s" % today(),
            "tags: [poseidon, tw_stock, %s, decision-loop, research]" % ", ".join('"%s"' % s for s in symbols),
            "origin: agent",
            'scope_key: "%s"' % scope_key,
            'revision_id: "%s"' % rev,
            "generated_at: %s" % now,
            "revision_status: completed",
            "as_of: %s" % i["as_of"],
            "stance: %s" % r["stance"],
            'manifest_sha256: "%s"' % i["manifest_sha256"],
            'request_sha256: "%s"' % i["request_sha256"],
            'content_sha256: "%s"' % i["content_sha256"],
            "previous_revision_id: %s"
            % ('"%s"' % i["previous_revision_id"] if i.get("previous_revision_id") else "null"),
            'policy_version: "%s"' % i["policy_version"],
            "provider: %s" % i["provider"],
            "model: %s" % i["model"],
            'poseidon_commit: "%s"' % a.poseidon_commit,
        ]
        if a.topic_source:
            fm.append('topic_source: "%s"' % a.topic_source.replace('"', "'"))
        fm += (
            [
                'ledger: "offline sqlite (not Poseidon production); evidence in %s"' % evidence_ref(a.evidence_dir),
                "related: [%s]" % ", ".join('"[[%s]]"' % x for x in related),
                "approval_link: null",
            ]
            + edge_lines(e)
            + ["---", ""]
        )
        note = [
            "> [!note] 來源與狀態",
            "> 用部署版 Poseidon（%s）的 research contract 在本機 SQLite ledger 走完 freeze → prepare → complete，尚未寫入 Poseidon production，故 `origin: agent`。"
            % a.poseidon_commit[:7],
        ]
        if i.get("previous_revision_id"):
            note.append("> 前一版：[[%s]]。" % i["previous_revision_id"])
        if a.note:
            note.append("> " + a.note)
        note.append("> 證據與可重放 ledger：`%s`。" % evidence_ref(a.evidence_dir))
        body = "\n".join(fm + note + ["", "# %s" % a.title, "", rendered])
        if not body.endswith("\n"):
            body += "\n"
        if target.exists():
            sys.exit("refusing to overwrite %s" % target)
        atomic_write(target, body + links_section(e), rev8)
        action = "projected"
    for s in create_stubs(vault, e, rev8):
        print("stub created: _kb/wiki/%s.md" % s)
    append_log(
        vault,
        "research | %s %s %s%s"
        % (scope_key.split(":")[-1], rev8, r["stance"], " (re-linked)" if action == "re-linked" else ""),
    )
    rl = vault / "_kb" / "ops" / "compile" / "research_links.py"
    if rl.exists():
        p = subprocess.run(
            [sys.executable, str(rl), "--apply"],
            env=dict(os.environ, NEXUS_VAULT=str(vault)),
            capture_output=True,
            text=True,
        )
        if p.returncode != 0:
            sys.exit("research_links.py failed: %s" % p.stderr.strip())
    else:
        print("note: %s not found; company blocks not regenerated" % rl, file=sys.stderr)
    print("%s %s" % (action, target))


if __name__ == "__main__":
    main()
