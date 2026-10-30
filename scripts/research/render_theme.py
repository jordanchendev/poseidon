#!/usr/bin/env python3
"""Render a research revision JSON (from research_theme.py) as Markdown. Sections that need fields a revision lacks are
omitted, so older revisions still render. Inference claims are prefixed 【推論】 (facts are not); a revision with
event_status pre_event opens with a warning banner (事前推論)."""

import json
import sys

from claim_precheck import evidence_tier as tier

d = json.load(open(sys.argv[1]))
i = d["identity"]
r = d["research"]
s = i["stats"]
syms = list(s)
ev = d.get("manifest_payload", {}).get("evidence", [])


def pct(x):
    return "n/a" if x is None else f"{x:+.2f}%"


bench = next(
    (
        e["payload"].get("benchmark")
        for e in d.get("manifest_payload", {}).get("evidence", [])
        if e.get("kind") == "price_facts" and e["payload"].get("benchmark")
    ),
    None,
)
banner = []
if r.get("event_status") == "pre_event":
    ed = r.get("event_date") or next(
        (c["date"] for c in sorted(r.get("catalysts") or [], key=lambda c: c["date"]) if c["date"] >= r["as_of"][:10]),
        None,
    )
    if ed and ed < r["as_of"][:10]:
        banner = [f"> [!warning] 事前推論：證據中沒有事件（{ed}）的結果，以下關於結果的內容皆為推論。\n"]
    else:
        banner = [f"> [!warning] 事前推論：事件（{ed or '日期未知'}）尚未發生，以下關於結果的內容皆為推論。\n"]
L = [
    *banner,
    f"**立場：{r['stance']}**　|　scope `{r['scope_key']}`　|　as_of `{r['as_of']}`　|　revision `{i['revision_id'][:8]}`　|　status `{i['status']}`\n",
    *(["> **推論｜一句話結論：** " + " ".join(r["tl_dr"].split()) + "\n"] if r.get("tl_dr") else []),
    "## 論點\n\n" + r["thesis"] + "\n",
    "## 價格證據快照（凍結於 manifest `" + i["manifest_sha256"][:12] + "…`）\n",
    "| 指標 | " + " | ".join(syms) + " |",
    "|---|" + "---|" * len(syms),
]
pf = next((e["payload"] for e in ev if e.get("kind") == "price_facts"), None)
ev_md = pf["event_time"][5:10] if pf else "事件前"
last_md = s[syms[0]]["last_bar"][5:10] if syms and s[syms[0]].get("last_bar") else "最新"
rows = [
    (f"收盤（{last_md}）", lambda x: f"{x['close']:g}"),
    (f"{ev_md} 收盤 → {last_md}", lambda x: f"{x['close_at_event']:g} → {pct(x['ret_since_event_pct'])}"),
    ("SMA5 / SMA20", lambda x: f"{x['sma5']:g} / {x['sma20']:g}"),
    (
        "最近 5/20 穿越",
        lambda x: f"{x['last_cross']['date'][5:10]} {'向上' if x['last_cross']['direction'] == 'up' else '向下'}",
    ),
    ("60 日高點（日期）", lambda x: f"{x['high_60d']:g}（{x['high_60d_date'][5:10]}）"),
    ("距高點", lambda x: pct(x["drawdown_from_60d_high_pct"])),
    ("5 日 / 20 日報酬", lambda x: f"{pct(x['ret_5d_pct'])} / {pct(x['ret_20d_pct'])}"),
    ("成交量 / 20 日均量", lambda x: f"{x['vol_last']:,.0f} / {x['vol_avg20']:,.0f}"),
    ("20 日平均日內振幅", lambda x: f"{x['range_avg20_pct']:.2f}%"),
]
if any("ret_t1_pct" in v for v in s.values()):
    rel = f"相對 {bench}" if bench else "相對基準"
    rows[2:2] = [
        ("T+1 / T+5 報酬（絕對）", lambda x: f"{pct(x.get('ret_t1_pct'))} / {pct(x.get('ret_t5_pct'))}"),
        (f"T+1 / T+5 報酬（{rel}）", lambda x: f"{pct(x.get('rel_ret_t1_pct'))} / {pct(x.get('rel_ret_t5_pct'))}"),
        (f"事件後至今（{rel}）", lambda x: pct(x.get("rel_ret_since_event_pct"))),
    ]
L += [f"| {name} | " + " | ".join(f(s[k]) for k in syms) + " |" for name, f in rows]
L.append("\n## 文件證據\n")
for e in i["evidence"]:
    if e["kind"] == "fundamental":
        L.append(f"- `{e['id']}`：{e['excerpts']} 段逐字節錄，event_time {e['event_time']}，sha `{e['sha256'][:12]}…`")
    elif e["kind"] == "secondary":
        L.append(
            f"- `{e['id']}`：二手素材 {e['excerpts']} 段逐字節錄（不作事實證據），event_time {e['event_time']}，sha `{e['sha256'][:12]}…`"
        )
    elif e["kind"] == "narrative_snapshot":
        L.append(
            f"- `{e['id']}`：敘事帳快照 {e.get('records')} 條（未展示 {e.get('not_shown')} 條），event_time {e['event_time']}，sha `{e['sha256'][:12]}…`"
        )
    elif e["kind"] == "price_facts":
        L.append(
            f"- `{e['id']}`：由凍結的 ohlcv 證據衍生（{e.get('algorithm')}），sha `{e['sha256'][:12]}…`；上表數字可由 manifest 重放"
        )
    elif e["kind"] == "kpi_series":
        L.append(
            f"- `{e['id']}`：TWSE OpenAPI 月營收 {e.get('points')} 個月，available {e['event_time']}，sha `{e['sha256'][:12]}…`"
        )
    elif e["kind"] == "consensus":
        L.append(
            f"- `{e['id']}`：{e.get('provider')} 共識 {e.get('items')} 項，observed {e.get('observed_at')}，sha `{e['sha256'][:12]}…`（third-party aggregate）"
        )
    elif e["kind"] == "surprise":
        L.append(
            f"- `{e['id']}`：由共識與實際值衍生（{e.get('algorithm')}），sha `{e['sha256'][:12]}…`；可由 manifest 重放"
        )


def num(x):
    return (
        "n/a"
        if x is None
        else (f"{int(x):,}" if float(x).is_integer() else f"{x:,.4g}" if abs(x) >= 1000 else f"{x:g}")
    )


def span(g):
    return (f"{num(g['low'])}–{num(g['high'])}" if "low" in g else num(g.get("value"))) if g else "n/a"


for k in (e for e in ev if e.get("kind") == "kpi_series"):
    p = k["payload"]
    pts = p["points"][-6:]
    L += [f"\n## 月營收（`{k['id']}`，單位：{p['unit']}）\n", "| 月份 | 營收 | YoY | MoM |", "|---|---|---|---|"]
    L += [f"| {x['period']} | {num(x['value'])} | {pct(x.get('yoy_pct'))} | {pct(x.get('mom_pct'))} |" for x in pts]
    t3 = p.get("trailing_3m_yoy_pct")
    L.append(
        f"\n近三個月合計 YoY：{pct(t3) if t3 is not None else 'n/a（需連續三個月資料）'}；來源 {p['source_uri']}，fetched_at {p.get('fetched_at') or 'unknown（離線檔）'}"
    )

guid = i.get("guidance") or []
br = i.get("guidance_bridge")
if br and br.get("rows"):
    L += [
        f"\n## Guidance bridge（前版 `{str(br['predecessor_revision_id'])[:8]}` 的 {'/'.join(br['reported_periods'])} 指引 → 實際 → 新指引）\n",
        "| 指標 | 前次指引 | 實際 | 實際 vs 前次 | 新指引 |",
        "|---|---|---|---|---|",
    ]
    for x in br["rows"]:
        act = (
            f"{num(x['actual']['value'])} {x['actual']['unit']}（{x['actual']['source']}）"
            if x["actual"]
            else "n/a（無引用實際值）"
        )
        new = "；".join(f"{n['period']}: {span(n)} {n.get('unit', '')}".strip() for n in x["new"]) or "n/a"
        L.append(
            f"| {x['metric']}（{x['period']}，{x.get('basis') or 'n/a'}） | {span(x['prior'])} {x['unit']} | {act} | {x['vs_prior']} | {new} |"
        )
if guid:
    L += [
        "\n## 本期指引（Guidance）\n",
        "| 指標 | 期間 | 區間 / 值 | 單位 | 基礎 | 匯率假設 | 節錄 |",
        "|---|---|---|---|---|---|---|",
    ]
    L += [
        f"| {g['metric']} | {g['period']} | {span(g)} | {g['unit']} | {g['basis']} | {g.get('fx_assumption') or '—'} | `{g['document_id']}` {g['excerpt_locator']} |"
        for g in guid
    ]


# ---- Consensus & surprise (gap B1) and variant perception (B5)
def cnum(x, unit):
    """Consensus numbers keep their unit; raw currency amounts >= 1e9 are shown in billions."""
    if x is None:
        return "n/a"
    if unit in ("USD", "TWD", "JPY") and abs(x) >= 1e9:
        return f"{x / 1e9:,.2f} billion {unit}"
    return f"{num(x)} {unit}".strip() if unit != "%" else f"{num(x)}%"


cons = [e for e in ev if e.get("kind") == "consensus"]
surp = {e["payload"]["consensus_evidence_id"]: e["payload"] for e in ev if e.get("kind") == "surprise"}
if cons or i.get("consensus_status"):
    L.append("\n## Consensus & surprise（第三方彙整共識；surprise 為衍生值）\n")
    for c in cons:
        p = c["payload"]
        rows_by = {(x["metric"], x["period"]): x for x in (surp.get(c["id"]) or {}).get("rows", [])}
        L += [
            f"`{c['id']}`：{p['provider']}{('（' + p['instrument'] + '）') if p.get('instrument') else ''}，observed {p['as_of']}，來源 {p['source_uri']}，raw sha `{p['raw_sha256'][:12]}…`\n",
            "| 指標 | 期間 | 共識均值（區間，n） | 實際 | Surprise | vs 區間 |",
            "|---|---|---|---|---|---|",
        ]
        for it in p["items"]:
            sr = rows_by.get((it["metric"], it["period"]))
            cell = f"{cnum(it['mean'], it['unit'])}（{cnum(it['low'], it['unit'])}–{cnum(it['high'], it['unit'])}，n={it['n_analysts']}）"
            if sr is None:
                act, sup, vs = "n/a", "n/a", "n/a"
            else:
                act = f"{num(sr['actual']['value'])} {sr['actual']['unit']}（{sr['actual']['source']}）"
                if sr.get("status") == "ok":
                    sp = f"{sr['surprise_pct']:+.2f}%" if sr.get("surprise_pct") is not None else "n/a"
                    sup, vs = (
                        f"{'+' if sr['surprise_abs'] >= 0 else '-'}{cnum(abs(sr['surprise_abs']), 'pp' if sr['unit'] == '%' else sr['unit'])}（{sp}）",
                        sr["vs_range"],
                    )
                else:
                    sup, vs = sr["status"], "n/a"
            L.append(f"| {it['metric']} | {it['period']} | {cell} | {act} | {sup} | {vs} |")
        L.append("")
    for st in i.get("consensus_status") or []:
        if st.get("status") != "ok":
            L.append(
                f"- 共識不可用：{st.get('symbol') or st.get('source')}"
                f"{(' via ' + st['ticker']) if st.get('ticker') else ''}：{st.get('status')}（{st.get('reason')}）"
            )
if r.get("variant_perception"):
    by_id = {c["id"]: c["payload"] for c in cons}
    L += [
        "\n## Variant perception（推論；我們 vs 共識）\n",
        "| 指標 | 期間 | 共識均值（引用） | 我們的看法 | 方向 | 為什麼 |",
        "|---|---|---|---|---|---|",
    ]
    for v in r["variant_perception"]:
        mk = lambda m: " ".join(str(m).replace("_", " ").lower().split())
        it = next(
            (
                x
                for x in by_id.get(v["consensus_ref"], {}).get("items", [])
                if mk(x["metric"]) == mk(v["metric"]) and x["period"] == v["period"]
            ),
            None,
        )
        mean = cnum(it["mean"], it["unit"]) if it else "n/a"
        ov = cnum(v["our_view"], it["unit"]) if it and isinstance(v["our_view"], (int, float)) else str(v["our_view"])
        L.append(
            f"| {v['metric']} | {v['period']} | {mean}（`{v['consensus_ref']}`） | {ov} | {v['direction']} | {v['why']} |"
        )

if i.get("read_across"):
    L += [
        f"\n## Read-across（推論；相對 {bench or '基準'} 的事件後報酬）\n",
        "| 代號 | 關係 | 為什麼 | T+1 相對 | T+5 相對 |",
        "|---|---|---|---|---|",
    ]
    for x in i["read_across"]:
        st = s.get(x["symbol"])
        t1, t5 = (pct(st.get("rel_ret_t1_pct")), pct(st.get("rel_ret_t5_pct"))) if st else ("n/a（未凍結價格）",) * 2
        L.append(f"| {x['symbol']} | {x['relation']} | {x['why']} | {t1} | {t5} |")
if r.get("catalysts"):
    L += ["\n## 催化劑日曆\n", "| 日期 | 類型 | 預期訊號 | 來源 |", "|---|---|---|---|"]
    L += [
        f"| {c['date']} | {c['type']} | {c['expected_signal']} | {c['source']} |"
        for c in sorted(r["catalysts"], key=lambda c: c["date"])
    ]
# inference claims carry a visible 【推論】 prefix wherever a human reads them (owner decision 2026-10-21); facts do not
for title, kind, mark in (("事實（引用凍結證據）", "fact", ""), ("推論", "inference", "【推論】")):
    L.append(f"\n## {title}\n")
    L += [f"- {mark}{c['text']}　_引用：{', '.join(c['evidence_ids'])}_" for c in r["claims"] if c["kind"] == kind] or [
        "- （無）"
    ]
for title, key in (("風險", "risks"), ("失效條件", "invalidation_conditions"), ("未知", "unknowns")):
    L.append(f"\n## {title}\n")
    L += [f"- {x}" for x in r[key]]
L += [
    "\n## 變更摘要\n\n" + r["change_summary"] + "\n",
    *(
        [f"**取代（supersedes）：** revision `{r['supersedes_revision_id']}`\n"]
        if r.get("supersedes_revision_id")
        else []
    ),
    "**下次檢視：** " + r["next_review_at"] + "\n",
]

# ---- 敘事變化 (design 2026-10-29 §4.5 step 5): from narrative_apply's results, never from model prose
STATUS_LABEL = {"partially_verified": "部分證實", "verified": "已證實", "refuted": "已否證", "stale": "過期未證實"}
TIER_LABEL = {"sell-side": "賣方", "media": "媒體", "kol": "KOL"}


def status_label(status, tier, claim_type):
    if status != "unverified":
        return STATUS_LABEL.get(status, status or "—")
    if tier in TIER_LABEL:
        return f"傳聞（{TIER_LABEL[tier]}）"
    if tier == "official":
        return "官方預期，未到驗證日" if claim_type == "forecast" else "官方說法，未證實"
    return "本系統推論，未證實" if tier == "our-inference" else "未證實"


def unwritten(x):
    """Why a rule outcome has no event: narrative_apply off, evidence that corroborated the record before, or a
    confirm-mode suggestion and where it stands."""
    sug = x.get("suggestion") or {}
    dup = x.get("duplicate_of") or sug.get("duplicate_of")
    if dup:
        return f"（同一份證據已背書過 {dup}，不再寫入）"
    if sug.get("status") == "suppressed":
        return f"（同一變化曾被駁回 {sug['suppressed_by']}，不再建議）"
    if sug.get("status") == "suggested":
        return f"（建議 {sug['id']}，待 owner 確認）"
    if sug.get("status") in ("confirmed", "rejected"):
        return f"（建議 {sug['id']} 已{'確認' if sug['status'] == 'confirmed' else '駁回'}）"
    return "（narrative_apply off，未寫入）"


def cell(text, n=None):
    text = " ".join(str(text).split()).replace("|", "／")
    return text if n is None or len(text) <= n else text[: n - 1] + "…"


snap = next((e["payload"] for e in ev if e.get("kind") == "narrative_snapshot"), None)
if snap and snap["records"]:
    nres = d.get("narrative_results") or {}
    res = {x["narrative_id"]: x for x in nres.get("records", [])}
    drafted = {x.get("narrative_id"): x for x in r.get("narrative_checks") or [] if isinstance(x, dict)}
    L += [
        "## 敘事變化\n",
        (
            f"narrative_apply `{nres.get('narrative_apply')}`；JEV `{nres.get('jev_model')}`；"
            f"展示 {len(snap['records'])} 條，未展示 {len(snap['not_shown'])} 條\n"
            if nres
            else "narrative_apply 未執行：下表只有 drafter 的判定，狀態未變。\n"
        ),
        "| narrative_id | claim | 來源 | 前狀態 | 判定 | 新狀態 | 依據（claim 編號與 quote） |",
        "|---|---|---|---|---|---|---|",
    ]
    owner, confirm = [], []
    for rec in snap["records"]:
        nid, x = rec["narrative_id"], res.get(rec["narrative_id"]) or {}
        src, ct = rec["source_tier"], rec["claim_type"]
        before = x.get("status_before", rec["status"])
        verdict = x.get("drafter_verdict") or drafted.get(nid, {}).get("verdict") or "—"
        judged = verdict + "".join(f"／{k} {x[f'{k}_label']}" for k in ("jev", "sweep") if x.get(f"{k}_label"))
        outcome, written = x.get("outcome"), x.get("event_seq") is not None
        if not x:
            new = "未判定"
        elif outcome == "status_change":
            new = status_label(x["target_status"], src, ct) + ("" if written else unwritten(x))
        elif outcome == "corroborated":
            new = status_label(before, src, ct) + "，官方背書" + ("" if written else unwritten(x))
        else:
            new = status_label(before, src, ct) + (f"（{outcome}）" if outcome in ("disputed", "missed") else "")
            new += f"（不變：{cell(x['why'])}）" if outcome == "no_change" and x.get("why") else ""
        idx = x.get("claim_indices", drafted.get(nid, {}).get("claim_indices")) or []
        basis = "；".join(
            f"claims[{i}]「{cell(r['claims'][i].get('quote', ''), 60)}」" for i in idx if 0 <= i < len(r["claims"])
        )
        L.append(
            f"| `{nid}` | {cell(rec['claim'])} | {cell(rec['attributed_to'])}（{src}） | "
            f"{status_label(before, src, ct)} | {judged} | {new} | {basis or '—'} |"
        )
        if (x.get("suggestion") or {}).get("status") == "suggested":
            confirm.append("- " + x["suggestion"]["line"])
        if outcome in ("disputed", "missed"):
            owner.append(
                f"- `{nid}` {outcome}（{x.get('rule')}）：{cell(rec['claim'])}；{x.get('why') or ''}".rstrip("；")
            )
    if confirm:
        L += ["\n### 待 owner 確認（narrative_apply confirm）\n", *confirm]
    if owner:
        L += ["\n### 待 owner 判斷\n", *owner]
    L.append("")
proposed = (d.get("narrative_results") or {}).get("narrative_proposals") or []
if proposed:
    STATUS_WORD = {"written": "已寫入", "held": "等待核准", "unfit": "未收"}
    L += ["## 新增敘事（本篇的 narrative_proposals）\n", "| narrative_id | claim | 結果 |", "|---|---|---|"]
    for x in proposed:
        waits = x.get("waits_text") or "、".join(x["waits"])  # a revision before waits_text lists the ids only
        why = f"：{cell(x['reason'])}" if x.get("reason") else (f"（等 {waits}）" if x["waits"] else "")
        L.append(f"| `{x['narrative_id']}` | {cell(x['claim'])} | {STATUS_WORD[x['status']]}{why} |")
    L.append("")
hits = (d.get("narrative_results") or {}).get("untagged_hits") or []
if hits:
    L += ["## 主題召回（關鍵字命中但未標主題，待 owner 判斷）\n"]
    L += [
        f"- claims[{h['claim_index']}]「{cell(r['claims'][h['claim_index']].get('text', ''), 60)}」命中 "
        f"`{h['theme_id']}` 的關鍵字「{h['keyword']}」"
        for h in hits
    ]
    L.append("")
L += [
    "## 可重放身分\n",
    "| 欄位 | 值 |\n|---|---|",
]
for k in (
    "manifest_id",
    "manifest_sha256",
    "revision_id",
    "request_sha256",
    "content_sha256",
    "runtime_digest",
    "previous_revision_id",
):
    L.append(f"| {k} | `{i[k]}` |")
L.append(f"| policy / provider / model | `{i['policy_version']}` / `{i['provider']}` / `{i['model']}` |")
for e in i["evidence"]:
    what = (
        f"{e['bars']} bars"
        if e["bars"]
        else (
            f"{e['excerpts']} excerpts"
            if e["excerpts"]
            else (
                f"{e['points']} monthly points"
                if e.get("points")
                else (f"{e['items']} consensus items" if e.get("items") else f"derived {e.get('algorithm')}")
            )
        )
    )
    L.append(f"| evidence {e['id']} | {what}，event_time {e['event_time']}，sha `{e['sha256'][:12]}…` |")
L.append(f"| replay | 同輸入重新 prepare → 同一 revision；改寫 thesis 後 complete → 被拒：`{i['mutation_refused']}` |")


# ---- Disclosure (gap C1): origin, identity, source tiers, verification summary
L += [
    "\n## 揭露（Disclosure）\n",
    f"- 來源：agent（origin: agent）；provider `{i['provider']}`，model `{i['model']}`，policy `{i['policy_version']}`",
]
disc = r.get("disclosure")
if isinstance(disc, dict) and disc.get("pilot_position"):
    L.append(
        f"- 部位揭露：pilot_position {disc['pilot_position']}"
        + (f"（{disc['size_note']}）" if disc.get("size_note") else "")
        + (f"，as_of {disc['as_of']}" if disc.get("as_of") else "")
    )
else:
    L.append(
        "- 部位揭露："
        + (
            ", ".join(f"{k}={v}" for k, v in disc.items())
            if isinstance(disc, dict)
            else "未記錄（revision 沒有 disclosure 欄位）"
        )
    )
L.append("- 證據來源與層級：")
L += [f"  - `{e['id']}`：{tier(e)}；{e.get('source_uri', '')}" for e in ev] or [
    "  - （manifest payload 不在此檔，無法列出）"
]
ver = i.get("verification") or {}
pc = i.get("claim_precheck") or []
cnt = lambda key, val: sum(1 for x in pc if x.get(key) == val)
L.append(
    f"- 節錄逐字驗證：{ver['excerpts_verified']}/{ver['excerpts_total']}"
    if ver
    else "- 節錄逐字驗證：未記錄（舊版 revision）"
)
L.append(
    f"- 數字 precheck：ok {cnt('numbers', 'ok')}，mismatch {cnt('numbers', 'mismatch')}，推論數字未對上（僅記錄） {cnt('numbers', 'mismatch-recorded')}，無數字 {cnt('numbers', 'no_numbers')}，略過 {cnt('numbers', 'skipped')}；原子化 ok {cnt('atomic', True)}/{len(pc)}"
    if pc
    else "- 數字 precheck：未記錄（舊版 revision）"
)
jev = d.get("jev_results") or r.get("jev_results")
if isinstance(jev, dict):
    keys = [k for k in ("model", "n", "supports", "contradicts", "says_nothing", "accuracy", "run_at") if k in jev]
    L.append("- JEV：" + ("，".join(f"{k} {jev[k]}" for k in keys) if keys else json.dumps(jev, ensure_ascii=False)))
else:
    L.append("- JEV：未執行（not run）")
L.append("- 本筆記由 agent 產生，非投資建議（research note generated by an agent; not investment advice）。")
print("\n".join(L))
