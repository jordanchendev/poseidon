"""Markdown rendering (zh-TW) for the four report kinds, frontmatter, file name, Telegram summary and Telegram text.

The archive md uses GFM tables for every numeric section (Nexus / Obsidian render them), change columns carry ▲/▼.
Each table is preceded by an HTML comment `<!-- tg: ... -->` naming the columns the phone version keeps.

Telegram text (`telegram_text`) is derived from the rendered markdown alone, so a delivered report can be re-sent
later. OpenClaw 2026.10.5 `message send` parses markdown with markdown-it (headingStyle "none": headings lose their
emphasis, so they become **bold** lines; GFM tables become <pre> blocks where CJK breaks monospace alignment, so each
table becomes our own fenced code block with ASCII-padded columns and the CJK label last) and chunks the rendered
HTML at 4000 characters. Its IR chunker re-renders each chunk, so a split inside a code block yields two valid <pre>
blocks, not broken markup; tables are kept short so that rarely happens.

The md is the complete record (Nexus archive and the file attached after the text). The phone text is the slim
version of the same md: tables trimmed per their tg options (see TG_* below), only the headlines the 解讀 cites as [n]
(numbers unchanged), no 資料來源, no `<!-- md-only -->` lines.
"""

from __future__ import annotations

import json
import re

from numfmt import fmt, fmt_pct, fmt_price, fmt_signed, fmt_signed_price

DISCLAIMER = "_本報告由程式彙整公開資料自動產生；數字皆來自資料源，解讀段落由模型依資料撰寫，非投資建議。_"
EXT_LABEL = {"PRE_MKT": "盤前", "POST_MKT": "盤後", "POST_MKT_PREV": "盤後"}

# Phone (Telegram) slimming. The archive md stays complete; each table's `<!-- tg: ... -->` comment may carry options after
# the column spec: `@head=N` first N rows, `@edge=N` first and last N rows, `@only=a,b` only those rows (by the label
# column, in that order), `@title=text` replaces the heading above the table. Lines prefixed `<!-- md-only -->` never
# reach the phone.
TG_EDGE = 5
TG_TOP = 10
TG_MOVERS = 3
TG_HEAD3 = "; @head=3"
MD_ONLY = "<!-- md-only -->"


def filename(kind, date):
    return f"{date}-{kind}.md"


def _q(s):
    return json.dumps(str(s), ensure_ascii=False)  # JSON string == valid YAML double-quoted scalar


def frontmatter(facts, model_id):
    market_tag = "tw_stock" if facts["market"] == "tw" else "us_stock"
    lines = [
        "---",
        "type: market_daily",
        f"title: {_q(facts['title'])}",
        f"created: {facts['generated_at'][:10]}",
        f"generated_at: {_q(facts['generated_at'])}",
        f"tags: [market_daily, {market_tag}, {facts['session']}]",
        f"market: {facts['market']}",
        f"session: {facts['session']}",
        f"session_date: {facts['date']}",
        f"as_of: {_q(facts['as_of'])}",
        f"model: {_q(model_id or 'none')}",
        "sources:",
    ]
    lines += [f"  - {_q(s['name'] + ' ' + s['url'])}" for s in facts["sources"]] or ["  []"]
    if facts["gaps"]:
        lines.append("data_gaps:")
        lines += [f"  - {_q(g)}" for g in facts["gaps"]]
    lines += ["promote: nexus", "---"]
    if not facts["sources"]:
        lines[lines.index("sources:")] = "sources: []"
        lines.remove("  []")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ table helpers
UP_COLOR = "#d6453d"  # TW convention in both markets: up = red, down = green
DOWN_COLOR = "#2e9e6a"  # contrast >= 3:1 on Obsidian light (#ffffff) and dark (#1e1e1e) backgrounds
_SPAN = re.compile(r"<span[^>]*>(.*?)</span>")


def _arw(x, text):
    """Change cell: ▲/▼ prefix inside a coloured span (Obsidian shows the colour; Kairos drops tags and colours by
    the ▲/▼ itself); flat and "—" stay plain."""
    if x is None or text == "—":
        return "—"
    if round(x, 4) == 0:
        return text
    color, arrow = (UP_COLOR, "▲") if x > 0 else (DOWN_COLOR, "▼")
    return f'<span style="color:{color}">{arrow}{text}</span>'


def chg_cell(x, nd=2):
    return _arw(x, fmt_signed(x, nd))


def pct_cell(x):
    return _arw(x, fmt_pct(x))


def pchg_cell(x):
    return _arw(x, fmt_signed_price(x))


NOT_FETCHED = "未取得"


def _or_na(cell, x, *args):
    """A change cell, or 「未取得」 when the value is missing (a gap, not a flat or unknown-by-design "—")."""
    return NOT_FETCHED if x is None else cell(x, *args)


def _table(headers, aligns, rows, tg=None):
    """GFM table. aligns: 'l' | 'r'. tg: Telegram spec 'md header=Label; ...; last header' (see telegram_text)."""
    out = [f"<!-- tg: {tg} -->"] if tg else []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join("---:" if a == "r" else "---" for a in aligns) + "|")
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c).replace("|", "／") for c in r) + " |")
    return out


def _quote_rows(rows, label_note=True):
    """Index / rate / FX rows -> [label, last, change, change%]; yields show bp in the change column."""
    out = []
    for r in rows:
        label = r["label"] + (f"（{r['note']}）" if label_note and r.get("note") else "")
        if r.get("missing"):
            out.append([label, "資料未取得", "—", "—"])
        elif r.get("kind") == "yield":
            bp = r.get("chg_bp")
            bp_cell = NOT_FETCHED if bp is None else _arw(bp, fmt_signed(bp, 1) + " bp")
            out.append([label, f"{fmt(r['last'], 3)}%", bp_cell, "—"])
        else:
            nd = r.get("nd", 2)
            out.append([label, fmt(r["last"], nd), _or_na(chg_cell, r["chg"], nd), _or_na(pct_cell, r["pct"])])
    return out


QUOTE_TG = "最新=Last; 漲跌=Chg!; 漲跌%=Chg%; 項目"


def _edge_tg(n, name, unit):
    """Phone options for a table sorted by change: top and bottom TG_EDGE rows, heading says so (only when it trims)."""
    if n <= 2 * TG_EDGE:
        return ""
    return f"; @edge={TG_EDGE}; @title={name}（{n} {unit}，漲跌前後各 {TG_EDGE}）"


def _quote_table(rows, first="項目", last_header="最新", tg_opts=""):
    tg = f"{last_header}=Last; 漲跌=Chg!; 漲跌%=Chg%; {first}{tg_opts}"
    return _table([first, last_header, "漲跌", "漲跌%"], ["l", "r", "r", "r"], _quote_rows(rows), tg=tg)


def _tw_watch_table(rows):
    body = []
    for w in rows:
        if w.get("missing"):
            body.append([w["code"], "—", "資料未取得", "—", "—", "—", w.get("note") or ""])
            continue
        body.append(
            [
                w["code"],
                w["name"],
                fmt_price(w["close"]),
                pchg_cell(w["chg"]),
                pct_cell(w["pct"]),
                fmt(w["turnover_yi"]),
                w.get("note") or "",
            ]
        )
    return _table(
        ["代號", "名稱", "收盤", "漲跌", "漲跌%", "成交值(億)", "備註"],
        ["l", "l", "r", "r", "r", "r", "l"],
        body,
        tg="代號=Code; 收盤=Close; 漲跌%=Chg%!; 名稱",
    )


def _us_watch_table(rows, ext_label):
    """ext_label 盤後 / 盤前: the extended-hours price and % columns."""
    body = []
    for r in rows:
        if r.get("missing"):
            body.append([r["key"], "資料未取得", "—", "—", "—", "—"])
            continue
        e = r.get("ext") or {}
        body.append(
            [
                r["key"],
                fmt(r["last"]),
                _or_na(chg_cell, r["chg"]),
                _or_na(pct_cell, r["pct"]),
                fmt(e["last"]) if e else "—",
                pct_cell(e.get("pct")) if e else "—",
            ]
        )
    tag = "AH%" if ext_label == "盤後" else "Pre%"
    return _table(
        ["代號", "收盤", "漲跌", "漲跌%", ext_label, f"{ext_label}%"],
        ["l", "r", "r", "r", "r", "r"],
        body,
        tg=f"代號=Sym; 收盤=Close; 漲跌%=Chg%!; {ext_label}%={tag}",
    )


def _kv_table(rows, value_header="數值"):
    return _table(["項目", value_header], ["l", "r"], rows, tg=f"{value_header}=Value; 項目")


def _inst_table(detail, nets):
    names = [
        ("外資及陸資（不含外資自營商）", "foreign"),
        ("投信", "trust"),
        ("自營商（自行買賣）", "dealer_self"),
        ("自營商（避險）", "dealer_hedge"),
        ("合計", "total"),
    ]
    if not detail and not nets:
        return ["資料未齊：三大法人買賣金額尚未公布"]
    body = []
    for label, k in names:
        dv = (detail or {}).get(k) or {}
        net = dv.get("net", (nets or {}).get(k))
        if net is None:
            continue
        body.append([label, fmt(dv.get("buy")) if dv else "—", fmt(dv.get("sell")) if dv else "—", chg_cell(net)])
    return _table(
        ["法人", "買進", "賣出", "買賣超"], ["l", "r", "r", "r"], body, tg="買賣超=Net!; 買進=Buy; 賣出=Sell; 法人"
    )


def _movers_table(items):
    if not items:
        return ["無"]
    body = [[s["code"], s["name"], pct_cell(s["pct"]), fmt(s["turnover_yi"])] for s in items]
    return _table(
        ["代號", "名稱", "漲跌%", "成交值(億)"],
        ["l", "l", "r", "r"],
        body,
        tg=f"代號=Code; 漲跌%=Chg%!; 成交值(億)=Amt; 名稱; @head={TG_MOVERS}",
    )


def _tw_summary_table(t):
    if not t:
        return ["資料未取得"]
    x = t["taiex"]
    rows = [
        ["加權指數", fmt(x["close"])],
        ["漲跌", chg_cell(x["chg"])],
        ["漲跌%", pct_cell(x["pct"])],
        ["集中市場成交金額（億）", fmt(t["turnover_total_yi"])],
    ]
    b = t.get("breadth_stocks") or {}
    if b.get("up") is not None:
        rows.append(["上市 上漲 / 下跌 / 持平", f"{b['up']} / {b['down']} / {b['flat']}"])
    inst = t.get("institutions_yi") or {}
    for label, k in (
        ("外資（億）", "foreign"),
        ("投信（億）", "trust"),
        ("自營商（億）", "dealer_total"),
        ("三大法人合計（億）", "total"),
    ):
        if inst.get(k) is not None:
            rows.append([label, chg_cell(inst[k])])
    return _kv_table(rows)


def _pct1_cell(x):
    return _arw(x, fmt_pct(x, 1))


GROUPS_SKIPPED = "族群強弱：本期未取得"  # shown in place of the section; the reason stays in the log and run record


def _group_section(g):
    """族群強弱 from the Thalassa group ranking (already in percent). All groups: 16 rows are ~0.6k characters on
    the phone, less than the 類股 table above it, so one table serves both the md and Telegram."""
    t = g["taiex"]
    out = [
        "",
        f"## 族群強弱（{len(g['groups'])} 族群，等權、已還原除權息）",
        "",
        f"加權指數 20 日 {fmt_pct(t['ret20'], 1)}、60 日 {fmt_pct(t['ret60'], 1)}。",
        "",
        MD_ONLY + "名次由 20 日、60 日相對加權指數的名次，加上成交占比變化的名次合計。",
        "",
    ]
    rows = [
        [
            x["rank"] if x["rank"] is not None else "—",
            x["name"],
            x["n"],
            _pct1_cell(x["ret20"]),
            _pct1_cell(x["ret60"]),
            fmt(x["share"], 1),
            _pct1_cell(x["share_chg"]),
        ]
        for x in g["groups"]
    ]
    out += _table(
        ["#", "族群", "檔數", "20日", "60日", "占比%", "占比變化"],
        ["r", "l", "r", "r", "r", "r", "r"],
        rows,
        tg="#=#; 20日=R20!; 60日=R60; 占比%=Shr; 占比變化=ShrChg; 族群",
    )
    out += ["", MD_ONLY + "占比＝族群近 20 日平均成交值占全市場普通股的比重；占比變化＝相對近 60 日平均。"]
    missing = [f"{x['name']} {'、'.join(x['missing'])}" for x in g["groups"] if x["missing"]]
    if missing:
        out += ["", "個股資料缺漏（未計入該族群）：" + "；".join(missing)]
    out += ["", "這是看資金流向的參考，不是買進訊號；族群輪動規則在樣本外並不穩定。"]
    return out


MARKET_ZH = {"twse": "上市", "tpex": "上櫃"}


def _credit_section(c):
    """信用交易 from the Thalassa market credit endpoint (already in display units). ``c`` is the shaped data or
    ``{"skipped": public reason}``. Changes are plain signed numbers: more margin is not good or bad news, so no
    red/green arrows."""
    if c.get("skipped"):
        return ["", f"信用交易：本期未取得（{c['skipped']}）"]
    out = ["", f"## 信用交易（{c['date']}，融資融券）", ""]
    rows = [
        [
            MARKET_ZH[m],
            fmt(x["amount_yi"], 1),
            fmt_signed(x["amount_chg_yi"], 1),
            fmt_signed(x["balance_chg"], 0),
            fmt_signed(x["short_chg"], 0),
            fmt(x["ratio"], 1) + "%",
        ]
        for m, x in c["latest"].items()
    ]
    out += _table(
        ["市場", "融資金額(億)", "金額增減(億)", "融資餘額增減(張)", "融券增減(張)", "估算維持率"],
        ["l", "r", "r", "r", "r", "r"],
        rows,
        tg="融資金額(億)=Amt; 金額增減(億)=AmtChg; 融資餘額增減(張)=BalChg; 融券增減(張)=ShChg; 估算維持率=Est; 市場",
    )
    out += [
        "",
        "估算維持率＝融資股票以當日收盤計的市值 ÷ 融資金額，是市場平均的估算值，不是交易所公布的數字。",
        "",
        MD_ONLY + "券商是對個別帳戶以整戶維持率 130% 追繳，所以這個數字不是斷頭線，看水位與變化即可。",
    ]
    if len(c["trend"]) > 1:
        t = [
            [
                r["date"][5:],
                fmt(r["twse"]["amount_yi"], 1),
                fmt(r["twse"]["ratio"], 1) + "%" if r["twse"]["ratio"] is not None else "—",
                fmt(r["tpex"]["amount_yi"], 1),
                fmt(r["tpex"]["ratio"], 1) + "%" if r["tpex"]["ratio"] is not None else "—",
            ]
            for r in c["trend"]
        ]
        out += ["", f"近 {len(t)} 個交易日（融資金額單位：億元）", ""]
        out += _table(
            ["日期", "上市融資", "上市估算", "上櫃融資", "上櫃估算"],
            ["l", "r", "r", "r", "r"],
            t,
            tg="上市融資=TWSE; 上市估算=TWEst; 上櫃融資=TPEx; 上櫃估算=TPEst; 日期",
        )
    return out


def _deterministic_one_liner(facts):
    d = facts["data"]
    k = facts["kind"]
    if k == "tw-postmarket":
        tx = next((i for i in d["indices"] if i["name"] == "加權指數"), None)
        b = d["breadth"].get("twse_stocks") or {}
        s = f"加權指數收 {fmt(tx['close'])}，{fmt_signed(tx['chg'])} 點（{fmt_pct(tx['pct'])}）"
        if b.get("up") is not None:
            s += f"；上市上漲 {b['up']} 家、下跌 {b['down']} 家"
        if d.get("institutions_yi"):
            s += f"；三大法人合計 {fmt_signed(d['institutions_yi'].get('total'))} 億元"
        return s + "。"
    if k == "tw-premarket":
        spx = next((r for r in d["us_indices"] if r["key"] == "SPX"), {})
        sox = next((r for r in d["us_indices"] if r["key"] == "SOX"), {})
        s = f"美股 {d['us_session_date']} S&P 500 {fmt_pct(spx.get('pct'))}、費半 {fmt_pct(sox.get('pct'))}"
        if d.get("tx_night"):
            s += f"；台指期夜盤 {fmt(d['tx_night']['last'], 0)}（{fmt_signed(d['tx_night']['chg'], 0)}）"
        if d.get("adr"):
            s += f"；TSM ADR {fmt_pct(d['adr']['adr_pct'])}"
        return s + "。"
    if k == "us-premarket":
        parts = [f"{r['label']} {fmt_pct(r['pct'])}" for r in d["futures"] if not r.get("missing")]
        s = "美股期貨：" + "、".join(parts)
        if d.get("tw_today"):
            s += f"；台股 {d['tw_today']['date']} 加權 {fmt_pct(d['tw_today']['taiex']['pct'])}"
        return s + "。"
    idx = {r["key"]: r for r in d["indices"]}
    parts = [
        f"{idx[k2]['label']} {fmt_pct(idx[k2]['pct'])}"
        for k2 in ("SPX", "IXIC", "DJI", "SOX")
        if k2 in idx and not idx[k2].get("missing")
    ]
    s = "美股收盤：" + "、".join(parts)
    if d["sectors"]:
        s += f"；類股最強 {d['sectors'][0]['label']} {fmt_pct(d['sectors'][0]['pct'])}、最弱 {d['sectors'][-1]['label']} {fmt_pct(d['sectors'][-1]['pct'])}"
    return s + "。"


# ------------------------------------------------------------------------------------------------ bodies
def _body_tw_post(d, cfg_tw, groups=None, groups_expected=False):
    out = ["## 大盤", ""]
    rows = [[i["name"], fmt(i["close"]), chg_cell(i["chg"]), pct_cell(i["pct"])] for i in d["indices"]]
    out += _table(
        ["指數", "收盤", "漲跌", "漲跌%"],
        ["l", "r", "r", "r"],
        rows,
        tg="收盤=Close; 漲跌=Chg; 漲跌%=Chg%!; 指數; @only=加權指數,櫃買指數,臺灣50",
    )
    t = d["turnover"]
    rows = [["一般股票", fmt(t["stocks_yi"])], ["集中市場總計", fmt(t["total_yi"])]]
    if t.get("avg5_total_yi") is not None:
        rows.append(["集中市場前 5 日均", fmt(t["avg5_total_yi"])])
    if t.get("otc_yi") is not None:
        rows.append(["上櫃", fmt(t["otc_yi"])])
    out += ["", "## 成交金額（億元）", "", *_kv_table(rows, "億元")]
    if t.get("vs_avg5_pct") is not None:
        out += ["", f"集中市場成交金額較前 5 日均 {pct_cell(t['vs_avg5_pct'])}"]
    out += ["", "## 漲跌結構", ""]
    b = d["breadth"]
    rows = []
    s_, a = b.get("twse_stocks") or {}, b.get("twse_all") or {}
    if s_:
        rows.append(["上市股票", s_["up"], s_["down"], s_["flat"], s_.get("up_limit") or 0, s_.get("down_limit") or 0])
    if a:
        rows.append(
            ["上市整體（含權證等）", a["up"], a["down"], a["flat"], a.get("up_limit") or 0, a.get("down_limit") or 0]
        )
    if b.get("tpex"):
        o = b["tpex"]
        rows.append(
            [
                "上櫃股票",
                int(o["up"]),
                int(o["down"]),
                int(o["flat"]),
                int(o["up_limit"] or 0),
                int(o["down_limit"] or 0),
            ]
        )
    elif b.get("tpex_computed"):
        o = b["tpex_computed"]
        rows.append(["上櫃普通股（依收盤行情計算）", o["up"], o["down"], o["flat"], "—", "—"])
    out += _table(
        ["市場", "上漲", "下跌", "持平", "漲停", "跌停"],
        ["l", "r", "r", "r", "r", "r"],
        rows,
        tg="上漲=Up; 下跌=Down; 持平=Flat; 漲停=LimUp; 跌停=LimDn; 市場",
    )
    secs = d["sectors"].get("all") or (d["sectors"]["leaders"] + d["sectors"]["laggards"])
    out += ["", f"## 類股表現（{len(secs)} 類，依漲跌排序）", ""]
    out += _table(
        ["類股", "收盤", "漲跌%"],
        ["l", "r", "r"],
        [[x["name"], fmt(x["close"]), pct_cell(x["pct"])] for x in secs],
        tg="漲跌%=Chg%!; 類股" + _edge_tg(len(secs), "類股表現", "類"),
    )
    if groups:
        out += _group_section(groups)
    elif groups_expected:
        out += ["", GROUPS_SKIPPED]
    out += [
        "",
        "## 三大法人（上市，單位：億元）",
        "",
        *_inst_table(d.get("institutions_detail_yi"), d["institutions_yi"]),
    ]
    out += ["", "## 成交金額前 20 大（上市普通股）", ""]
    rows = [
        [n, s["code"], s["name"], fmt(s["turnover_yi"]), fmt_price(s["close"]), pchg_cell(s["chg"]), pct_cell(s["pct"])]
        for n, s in enumerate(d["top_turnover"], 1)
    ]
    out += _table(
        ["#", "代號", "名稱", "成交值(億)", "收盤", "漲跌", "漲跌%"],
        ["r", "l", "l", "r", "r", "r", "r"],
        rows,
        tg=f"#=#; 代號=Code; 成交值(億)=Amt; 漲跌%=Chg%!; 名稱; @head={TG_TOP}; @title=成交金額前 {TG_TOP} 大（上市普通股）",
    )
    m = d["movers"]
    out += ["", "## 焦點個股", ""]
    out += [f"### 上市漲幅（成交 ≥ {m['threshold_yi']} 億）", "", *_movers_table(m["gainers"])]
    out += ["", f"### 上市跌幅（成交 ≥ {m['threshold_yi']} 億）", "", *_movers_table(m["losers"])]
    om = d.get("otc_movers")
    if om:
        out += ["", f"### 上櫃漲幅（成交 ≥ {om['threshold_yi']} 億）", "", *_movers_table(om["gainers"])]
        out += ["", f"### 上櫃跌幅（成交 ≥ {om['threshold_yi']} 億）", "", *_movers_table(om["losers"])]
    out += ["", "## Watchlist", "", *_tw_watch_table(d["watchlist"])]
    return out


def _body_tw_pre(d, credit=None):
    out = [f"## 美股收盤（{d['us_session_date']}）", "", *_quote_table(d["us_indices"], "指數", "收盤", TG_HEAD3)]
    out += ["", "## 利率、匯率與商品（產出時最新報價）", "", *_quote_table(d["rates_fx"])]
    out += ["", "## 台積電 ADR", ""]
    a = d.get("adr")
    if a:
        out += _kv_table(
            [
                [f"{a['adr']} 收盤（美元）", fmt(a["adr_close"])],
                [f"{a['adr']} 漲跌%", pct_cell(a["adr_pct"])],
                ["美元/新台幣", fmt(a["usdtwd"], 3)],
                [f"折合台股（1 ADR = {a['shares_per_adr']} 股，元）", fmt(a["equiv_twd"])],
                [f"{a['tw_code']} {a['tw_date']} 收盤（元）", fmt_price(a["tw_close"])],
                ["ADR 溢價%", pct_cell(a["premium_pct"])],
            ]
        )
    else:
        out.append("資料未齊")
    out += ["", "## 台指期夜盤（近月）", ""]
    n = d.get("tx_night")
    if n:
        out += _table(
            ["合約", "最後", "漲跌", "漲跌%", "最高", "最低", "成交量"],
            ["l", "r", "r", "r", "r", "r", "r"],
            [
                [
                    n["name"],
                    fmt(n["last"], 0),
                    chg_cell(n["chg"], 0),
                    pct_cell(n["pct"]),
                    fmt(n["high"], 0),
                    fmt(n["low"], 0),
                    fmt(n["volume"], 0),
                ]
            ],
            tg="最後=Last; 漲跌=Chg!; 漲跌%=Chg%; 成交量=Vol; 合約",
        )
        extra = f"參考價 {fmt(n['ref'], 0)}（{n.get('source', 'TAIFEX')}）"
        if n.get("basis_vs_taiex") is not None:
            extra += f"；相對前一交易日加權指數收盤 {chg_cell(n['basis_vs_taiex'], 0)} 點"
        out += ["", extra]
    else:
        out.append("資料未齊：夜盤行情未取得")
    prev = d.get("prev_tw")
    out += ["", f"## 前一交易日台股（{prev['date']}）" if prev else "## 前一交易日台股", "", *_tw_summary_table(prev)]
    if prev:
        out += ["", "### Watchlist（前一交易日收盤）", "", *_tw_watch_table(prev["watchlist"])]
    if credit:
        out += _credit_section(credit)
    out += ["", "## 今日除權息（上市）", ""]
    ex = d.get("exdiv_today")
    if ex is None:
        out.append("資料未齊")
    elif not ex:
        out.append("無")
    else:
        rows = []
        for e in ex:
            cash = fmt(e["cash"], 4).rstrip("0").rstrip(".") if e.get("cash") else (e.get("cash_text") or "—")
            rows.append([e["code"], e["name"], e["kind"], cash, e.get("stock_ratio") or "—"])
        out += _table(
            ["代號", "名稱", "除權息", "現金股利(元)", "配股率"],
            ["l", "l", "l", "r", "r"],
            rows,
            tg="代號=Code; 現金股利(元)=Cash; 名稱",
        )
    out += ["", f"## 美股 Watchlist（{d['us_session_date']} 收盤）", "", *_us_watch_table(d["us_watchlist"], "盤後")]
    return out


def _body_us_pre(d):
    rows = list(d["futures"]) + [dict(r, label="VIX（最新）") for r in d["vix_prev"]]
    out = ["## 指數期貨與 VIX", "", *_quote_table(rows, "項目", "最新", TG_HEAD3)]
    out += ["", "## 利率、匯率與商品（產出時最新報價）", "", *_quote_table(d["rates_fx"])]
    out += ["", "## 亞股收盤", "", *_quote_table(d["asia"], "指數", "收盤")]
    tw = d.get("tw_today")
    out += ["", f"## 今日台股（{tw['date']}）" if tw else "## 今日台股", "", *_tw_summary_table(tw)]
    if tw:
        out += ["", "### 台股 Watchlist", "", *_tw_watch_table(tw["watchlist"])]
    out += ["", "## 美股 Watchlist 盤前（收盤欄為前一交易日）", "", *_us_watch_table(d["us_watchlist_pre"], "盤前")]
    return out


def _body_us_post(d):
    out = ["## 指數", "", *_quote_table(d["indices"], "指數", "收盤", TG_HEAD3)]
    out += ["", "## 利率、匯率與商品（產出時最新報價）", "", *_quote_table(d["rates_fx"])]
    out += ["", "## 類股 ETF（依漲跌排序）", ""]
    rows = [[r["key"], r["label"], fmt(r["last"]), chg_cell(r["chg"]), pct_cell(r["pct"])] for r in d["sectors"]]
    out += _table(
        ["ETF", "類股", "收盤", "漲跌", "漲跌%"],
        ["l", "l", "r", "r", "r"],
        rows,
        tg="ETF=ETF; 漲跌%=Chg%!; 收盤=Close; 類股" + _edge_tg(len(rows), "類股 ETF", "檔"),
    )
    out += ["", "## Watchlist（收盤與盤後）", "", *_us_watch_table(d["watchlist"], "盤後")]
    a = d.get("adr")
    out += ["", "## 台積電 ADR 對台股", ""]
    if a:
        out += _kv_table(
            [
                [f"{a['adr']} 收盤（美元）", fmt(a["adr_close"])],
                [f"{a['adr']} 漲跌%", pct_cell(a["adr_pct"])],
                ["美元/新台幣", fmt(a["usdtwd"], 3)],
                [f"折合台股（1 ADR = {a['shares_per_adr']} 股，元）", fmt(a["equiv_twd"])],
                [f"{a['tw_code']} {a['tw_date']} 收盤（元）", fmt_price(a["tw_close"])],
                ["ADR 溢價%", pct_cell(a["premium_pct"])],
            ]
        )
    else:
        out.append("資料未齊")
    return out


def render(facts, section=None, llm_meta=None, cfg=None):
    """-> (markdown, telegram summary)."""
    d = facts["data"]
    kind = facts["kind"]
    model_id = (llm_meta or {}).get("returned_model") if section else None
    det_one = _deterministic_one_liner(facts)
    parts = [frontmatter(facts, model_id), "", f"# {facts['title']}", ""]
    if facts["gaps"]:
        parts += ["> **資料未齊**：" + "；".join(facts["gaps"]), ""]
    parts += ["## 一句話", section["one_liner"] if section else det_one, ""]
    body = {
        "tw-postmarket": lambda: _body_tw_post(
            d, (cfg or {}).get("tw", {}), facts.get("group_ranking"), bool((cfg or {}).get("thalassa"))
        ),
        "tw-premarket": lambda: _body_tw_pre(d, facts.get("credit")),
        "us-premarket": lambda: _body_us_pre(d),
        "us-postmarket": lambda: _body_us_post(d),
    }[kind]()
    parts += body
    parts += ["", "## 解讀（非投資建議）"]
    if section:
        parts += [section["interpretation"], "", "觀察重點："] + [f"- {w}" for w in section["watch_points"]]
        if (llm_meta or {}).get("dropped"):
            parts += ["", f"_（已移除 {len(llm_meta['dropped'])} 句資料中沒有依據的技術面描述）_"]
    else:
        status = (llm_meta or {}).get("status")
        if status in ("refused", "error"):
            parts.append("_AI 解讀未通過資料核對或呼叫失敗，本期略過（數據段落不受影響）。_")
        else:
            parts.append("_（未啟用 AI 解讀：--no-llm）_")
    if facts["headlines"]:
        parts += ["", "## 新聞標題（解讀引用編號）"]
        parts += [
            f"{h['n']}. [{h['title']}]({h['link']}) — {h['source']}，{h['published']}" for h in facts["headlines"]
        ]
    parts += ["", "## 資料來源"] + [
        f"- [{s['name'].replace('[', '［').replace(']', '］')}]({s['url']})" for s in facts["sources"]
    ]
    parts += ["", DISCLAIMER, ""]
    md = "\n".join(parts)
    summary_lines = [facts["title"], det_one]
    if section:
        summary_lines.append("一句話：" + section["one_liner"])
    if facts["gaps"]:
        summary_lines.append("資料未齊：" + "；".join(facts["gaps"])[:150])
    summary = "\n".join(summary_lines)[:580]
    return md, summary


# ------------------------------------------------------------------------------------------------ Telegram text
TELEGRAM_TEXT_CAP = 19000  # the bridge refuses > 20000
_HEADING = re.compile(r"^#{1,6}\s+(.*)$")
_NEWS = re.compile(r"^(\d+)\. \[(.*)\]\((\S+)\) — (.+?)(?:，\d{4}-\d{2}-\d{2}T\S+)?$")
_LONE_STAR = re.compile(r"(?<!\*)\*(?!\*)")


def _cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _table_row_line(header, cells):
    cells = cells + [""] * (len(header) - len(cells))
    if header[:2] == ["代號", "名稱"]:  # TW watchlist
        code, name, close, chg, pct, turnover, note = (cells + [""] * 7)[:7]
        if close == "資料未取得":
            return f"{code} 資料未取得" + (f" · {note}" if note else "")
        line = f"{code} {name} {close} {chg}（{pct}）成交 {turnover} 億"
        return line + (f" · {note}" if note else "")
    if header[:2] == ["代號", "收盤"]:  # US watchlist (close + pre/after-hours)
        sym, close, chg, pct, ext = (cells + [""] * 5)[:5]
        if close == "資料未取得":
            return f"{sym} 資料未取得"
        line = f"{sym} {close} {chg}（{pct}）"
        return line + (f" · {ext}" if ext and ext != "—" else "")
    return " · ".join(c for c in cells if c)


def _tg_spec(text):
    """'收盤=Close; 漲跌%=Chg%!; 指數' -> [(md header, label, primary)]; the last item is the unpadded (CJK) column;
    a trailing '!' marks the column whose ▲/▼ picks the row's 🔴/🟢 marker."""
    out = []
    for item in text.split(";"):
        item = item.strip()
        if item and not item.startswith("@"):
            h, _, label = item.partition("=")
            label = (label or h).strip()
            primary = label.endswith("!")
            out.append((h.strip(), label.rstrip("!"), primary))
    return out


def _tg_opts(text):
    """'a=A; @head=3; @title=x' -> {'head': '3', 'title': 'x'}: the slimming options of a tg comment."""
    out = {}
    for item in text.split(";"):
        item = item.strip()
        if item.startswith("@"):
            k, _, v = item[1:].partition("=")
            out[k.strip()] = v.strip()
    return out


def _trim_rows(rows, label_col, opts):
    """Apply @only / @head / @edge to data rows -> (rows, gap) where gap = (position, omitted) for @edge."""
    only = opts.get("only")
    if only:
        by = {(r[label_col] if label_col < len(r) else "").strip(): r for r in rows}
        picked = [by[n.strip()] for n in only.split(",") if n.strip() in by]
        if picked:
            return picked, None
    if opts.get("head", "").isdigit() and len(rows) > int(opts["head"]):
        return rows[: int(opts["head"])], None
    n = int(opts["edge"]) if opts.get("edge", "").isdigit() else 0
    if n and len(rows) > 2 * n:
        return rows[:n] + rows[-n:], (n, len(rows) - 2 * n)
    return rows, None


_NUMERIC_CELL = re.compile(r"^(?:[+-]?[\d,]*\.?\d+(?:%| bp)?|-|--)$")


def _ascii_cell(c):
    c = c.replace("▲", "").replace("▼", "").replace("—", "-").replace("−", "-").strip()
    c = "".join(ch for ch in c if ord(ch) < 128).strip()
    return c or "-"


def code_table(header, rows, spec, max_width=34, opts=None):
    """A GFM table as a fenced code block for Telegram (OpenClaw renders tables as <pre> anyway, where CJK breaks
    alignment): only the spec's columns, ASCII cells padded (numbers right-aligned), the CJK column last and unpadded."""
    spec = [x if len(x) == 3 else (x[0], x[1], False) for x in spec]
    cols = [(header.index(h), label) for h, label, _p in spec if h in header]
    prim = next((header.index(h) for h, _l, p in spec if p and h in header), None)
    if len(cols) < 2:
        return None
    rows, gap = _trim_rows(rows, cols[-1][0], opts or {})
    body = [[r[i] if i < len(r) else "" for i, _ in cols] for r in rows]
    fixed = [[label for _, label in cols[:-1]]] + [[_ascii_cell(c) for c in r[:-1]] for r in body]
    last = [cols[-1][1]] + [r[-1].replace("▲", "").replace("▼", "").strip() or "-" for r in body]
    if all(_NUMERIC_CELL.match(x.replace("—", "-").replace("−", "-").strip()) for x in last[1:]):
        last = [last[0]] + [_ascii_cell(x) for x in last[1:]]  # an all-numeric last column (e.g. AH%) stays ASCII
    n = len(cols) - 1
    widths = [max(len(r[j]) for r in fixed) for j in range(n)]
    right = [sum(1 for r in fixed[1:] if _NUMERIC_CELL.match(r[j])) * 2 >= len(fixed) - 1 for j in range(n)]
    sep = "  " if sum(widths) + 2 * n <= max_width else " "
    marks = [""] + [_row_mark(r[prim] if prim is not None and prim < len(r) else "") for r in rows]
    lines = []
    for r, tail, mark in zip(fixed, last, marks, strict=True):
        cells = [r[j].rjust(widths[j]) if right[j] else r[j].ljust(widths[j]) for j in range(n)]
        lines.append((sep.join(cells) + " " + tail).rstrip() + mark)
    if gap:
        lines.insert(1 + gap[0], f"... {gap[1]} omitted")
    return ["```", *lines, "```"]


def _row_mark(cell):
    """' 🔴' for ▲ (up), ' 🟢' for ▼ (down), '' otherwise; appended at the very end so alignment is unchanged."""
    return " 🔴" if "▲" in cell else " 🟢" if "▼" in cell else ""


_CITE = re.compile(r"\[(\d{1,3})\](?!\()")


def _cited(lines):
    """Headline numbers the report text refers to as [n] (outside tables and the news list itself)."""
    return {int(n) for x in lines if not x.startswith("|") and not _NEWS.match(x) for n in _CITE.findall(x)}


def telegram_text(md):
    """Report markdown -> Telegram markdown: no frontmatter / 資料來源 / HTML comments, headings as bold lines, news as
    numbered links, and each GFM table as a fenced code block per its `<!-- tg: ... -->` spec (ASCII-padded numeric
    columns, CJK label last). Tables without a spec (reports rendered before specs existed) become one line per row.
    Everything else (banner, 一句話, 解讀, 觀察重點) is kept verbatim."""
    m = re.match(r"---\n.*?\n---\n?", md, re.S)
    lines = _SPAN.sub(r"\1", md[m.end() :] if m else md).splitlines()
    cited = _cited(lines)
    out, skipping, spec, opts, i, news_head = [], False, None, {}, 0, None
    while i < len(lines):
        line = lines[i].rstrip()
        i += 1
        h = _HEADING.match(line)
        if h:
            skipping = h.group(1).startswith("資料來源")
            if not skipping:
                out.append(f"**{h.group(1).strip()}**")
            news_head = len(out) - 1 if h.group(1).startswith("新聞標題") else news_head
            continue
        if skipping:
            if not line.startswith("_本報告"):
                continue
            skipping = False
        tgm = re.match(r"^<!--\s*tg:\s*(.*?)\s*-->$", line)
        if tgm:
            spec, opts = _tg_spec(tgm.group(1)), _tg_opts(tgm.group(1))
            title = opts.get("title")
            if title:
                k = next((j for j in range(len(out) - 1, -1, -1) if out[j].startswith("**")), None)
                if k is not None:
                    out[k] = f"**{title}**"
            continue
        if line.startswith("<!--"):
            continue
        if line.startswith("|"):
            block = [line]
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i].rstrip())
                i += 1
            header = _cells(block[0])
            rows = [_cells(b) for b in block[1:] if not all(set(c) <= set("-:") for c in _cells(b))]
            rendered = code_table(header, rows, spec, opts=opts) if spec else None
            if rendered:
                out += rendered
            else:
                rows = _trim_rows(rows, len(header) - 1, opts)[0]
                out += [_LONE_STAR.sub(r"\\*", _table_row_line(header, r)) for r in rows]
            spec, opts = None, {}
            continue
        if line.startswith("> "):
            line = line[2:]
        n = _NEWS.match(line)
        if n:
            if int(n.group(1)) not in cited:
                continue
            title = n.group(2).replace("[", "［").replace("]", "］")
            out.append(f"{n.group(1)}. [{title}]({n.group(3)}) — {n.group(4)}")
            continue
        if "**" not in line:
            line = _LONE_STAR.sub(r"\\*", line)
        out.append(line)
    if news_head is not None and not any(_NEWS.match(x) for x in out):
        del out[news_head]  # no headline is cited: the heading goes too
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"
    if len(text) > TELEGRAM_TEXT_CAP:
        text = text[: TELEGRAM_TEXT_CAP - 40].rstrip() + "\n…（內容過長已截斷，完整內容見 md 附件）\n"
    return text
