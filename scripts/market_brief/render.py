"""Markdown rendering (zh-TW) for the four report kinds, frontmatter, file name and Telegram summary."""

from __future__ import annotations

import json

from numfmt import fmt, fmt_pct, fmt_price, fmt_signed, fmt_signed_price

DISCLAIMER = "_本報告由程式彙整公開資料自動產生；數字皆來自資料源，解讀段落由模型依資料撰寫，非投資建議。_"
EXT_LABEL = {"PRE_MKT": "盤前", "POST_MKT": "盤後", "POST_MKT_PREV": "盤後"}


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


def _chg_line(label, r):
    if r is not None and r.get("note"):
        label = f"{label}（{r['note']}）"
    if r is None or r.get("missing"):
        return f"- {label}：資料未取得"
    if r.get("kind") == "yield":
        return f"- {label}：{fmt(r['last'], 3)}%（{fmt_signed(r.get('chg_bp'), 1)} bp）"
    nd = r.get("nd", 2)
    return f"- {label}：{fmt(r['last'], nd)}，{fmt_signed(r['chg'], nd)}（{fmt_pct(r['pct'])}）"


def _ext_text(r):
    e = r.get("ext")
    if not e:
        return "—"
    return f"{EXT_LABEL.get(e['type'], e['type'])} {fmt(e['last'])}（{fmt_pct(e['pct'])}）"


def _stock(s):
    return f"{s['code']} {s['name']}"


def _tw_watch_table(rows):
    out = ["| 代號 | 名稱 | 收盤 | 漲跌 | 漲跌% | 成交值(億) | 備註 |", "|---|---|---:|---:|---:|---:|---|"]
    for w in rows:
        if w.get("missing"):
            out.append(f"| {w['code']} | — | 資料未取得 | | | | {w.get('note') or ''} |")
            continue
        out.append(
            f"| {w['code']} | {w['name']} | {fmt_price(w['close'])} | {fmt_signed_price(w['chg'])} | {fmt_pct(w['pct'])} | "
            f"{fmt(w['turnover_yi'])} | {w.get('note') or ''} |"
        )
    return out


def _us_watch_table(rows, ext_header):
    out = [f"| 代號 | 收盤 | 漲跌 | 漲跌% | {ext_header} |", "|---|---:|---:|---:|---|"]
    for r in rows:
        if r.get("missing"):
            out.append(f"| {r['key']} | 資料未取得 | | | |")
            continue
        out.append(f"| {r['key']} | {fmt(r['last'])} | {fmt_signed(r['chg'])} | {fmt_pct(r['pct'])} | {_ext_text(r)} |")
    return out


def _inst_lines(inst):
    if not inst:
        return ["- 資料未齊：三大法人買賣金額尚未公布"]
    rows = [
        ("外資及陸資（不含外資自營商）", "foreign"),
        ("投信", "trust"),
        ("自營商（自行買賣）", "dealer_self"),
        ("自營商（避險）", "dealer_hedge"),
        ("合計", "total"),
    ]
    return [f"- {label}：{fmt_signed(inst.get(k))}" for label, k in rows if inst.get(k) is not None]


def _movers_line(items):
    return "、".join(f"{s['name']} {s['code']} {fmt_pct(s['pct'])}" for s in items) or "無"


def _tw_summary_lines(t, heading_date_label):
    if not t:
        return ["- 資料未取得"]
    x = t["taiex"]
    out = [
        f"- {heading_date_label}（{t['date']}）加權指數：{fmt(x['close'])}，{fmt_signed(x['chg'])}（{fmt_pct(x['pct'])}）",
        f"- 集中市場成交金額：{fmt(t['turnover_total_yi'])} 億元",
    ]
    b = t.get("breadth_stocks") or {}
    if b.get("up") is not None:
        out.append(f"- 上市股票：上漲 {b['up']}、下跌 {b['down']}、持平 {b['flat']}")
    inst = t.get("institutions_yi")
    if inst:
        out.append(
            f"- 三大法人（億元）：外資 {fmt_signed(inst.get('foreign'))}、投信 {fmt_signed(inst.get('trust'))}、"
            f"自營商 {fmt_signed(inst.get('dealer_total'))}、合計 {fmt_signed(inst.get('total'))}"
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
def _body_tw_post(d, cfg_tw):
    out = ["## 大盤"]
    for i in d["indices"]:
        out.append(f"- {i['name']}：{fmt(i['close'])}，{fmt_signed(i['chg'])}（{fmt_pct(i['pct'])}）")
    t = d["turnover"]
    line = f"- 成交金額：一般股票 {fmt(t['stocks_yi'])} 億；集中市場總計 {fmt(t['total_yi'])} 億"
    if t.get("vs_avg5_pct") is not None:
        line += f"（前 5 日均 {fmt(t['avg5_total_yi'])} 億，{fmt_pct(t['vs_avg5_pct'])}）"
    if t.get("otc_yi") is not None:
        line += f"；上櫃 {fmt(t['otc_yi'])} 億"
    out += [line, "", "## 漲跌結構"]
    b = d["breadth"]
    s, a = b.get("twse_stocks") or {}, b.get("twse_all") or {}
    if s:
        out.append(
            f"- 上市股票：上漲 {s['up']}（漲停 {s.get('up_limit') or 0}）、下跌 {s['down']}（跌停 {s.get('down_limit') or 0}）、持平 {s['flat']}"
        )
    if a:
        out.append(f"- 上市整體市場（含權證等）：上漲 {a['up']}、下跌 {a['down']}、持平 {a['flat']}")
    if b.get("tpex"):
        o = b["tpex"]
        out.append(
            f"- 上櫃股票：上漲 {int(o['up'])}（漲停 {int(o['up_limit'] or 0)}）、下跌 {int(o['down'])}（跌停 {int(o['down_limit'] or 0)}）、持平 {int(o['flat'])}"
        )
    elif b.get("tpex_computed"):
        o = b["tpex_computed"]
        out.append(
            f"- 上櫃普通股（依收盤行情計算，官方彙總未公布）：上漲 {o['up']}、下跌 {o['down']}、持平 {o['flat']}"
        )
    else:
        out.append("- 上櫃：資料未齊")
    out += ["", "## 類股表現（漲跌 %）", "領漲："]
    out += [f"- {x['name']} {fmt_pct(x['pct'])}" for x in d["sectors"]["leaders"]]
    out.append("")
    out.append("落後：")
    out += [f"- {x['name']} {fmt_pct(x['pct'])}" for x in d["sectors"]["laggards"]]
    out += ["", "## 三大法人（上市，單位：億元）", *_inst_lines(d["institutions_yi"])]
    out += ["", "## 成交金額前 20 大（億元｜收盤｜漲跌）"]
    for n, s in enumerate(d["top_turnover"], 1):
        out.append(
            f"{n}. {_stock(s)} {fmt(s['turnover_yi'])}｜{fmt_price(s['close'])}（{fmt_signed_price(s['chg'])} / {fmt_pct(s['pct'])}）"
        )
    m = d["movers"]
    out += [
        "",
        "## 焦點個股",
        f"- 上市漲幅（成交 ≥ {m['threshold_yi']} 億）：{_movers_line(m['gainers'])}",
        f"- 上市跌幅（成交 ≥ {m['threshold_yi']} 億）：{_movers_line(m['losers'])}",
    ]
    om = d.get("otc_movers")
    if om:
        out += [
            f"- 上櫃漲幅（成交 ≥ {om['threshold_yi']} 億）：{_movers_line(om['gainers'])}",
            f"- 上櫃跌幅（成交 ≥ {om['threshold_yi']} 億）：{_movers_line(om['losers'])}",
        ]
    out += ["", "## Watchlist", *_tw_watch_table(d["watchlist"])]
    return out


def _body_tw_pre(d):
    out = [f"## 美股收盤（{d['us_session_date']}）"]
    out += [_chg_line(r["label"], r) for r in d["us_indices"]]
    out += ["", "## 利率、匯率與商品（產出時最新報價）"] + [_chg_line(r["label"], r) for r in d["rates_fx"]]
    out += ["", "## 台積電 ADR"]
    a = d.get("adr")
    if a:
        out += [
            f"- {a['adr']}：{fmt(a['adr_close'])} 美元（{fmt_pct(a['adr_pct'])}）",
            f"- 折合台股：{fmt(a['equiv_twd'])} 元（匯率 {fmt(a['usdtwd'], 3)}，1 ADR = {a['shares_per_adr']} 股）",
            f"- 相對 {a['tw_code']} {a['tw_date']} 收盤 {fmt_price(a['tw_close'])} 元：溢價 {fmt_pct(a['premium_pct'])}",
        ]
    else:
        out.append("- 資料未齊")
    out += ["", "## 台指期夜盤（近月）"]
    n = d.get("tx_night")
    if n:
        out.append(
            f"- {n['name']}：{fmt(n['last'], 0)}，{fmt_signed(n['chg'], 0)}（{fmt_pct(n['pct'])}），參考價 {fmt(n['ref'], 0)}（{n.get('source', 'TAIFEX')}）"
        )
        out.append(f"- 高 {fmt(n['high'], 0)}／低 {fmt(n['low'], 0)}，成交量 {fmt(n['volume'], 0)} 口")
        if n.get("basis_vs_taiex") is not None:
            out.append(f"- 相對前一交易日加權指數收盤：{fmt_signed(n['basis_vs_taiex'], 0)} 點")
    else:
        out.append("- 資料未齊：夜盤行情未取得")
    out += ["", "## 前一交易日台股", *_tw_summary_lines(d.get("prev_tw"), "前一交易日")]
    if d.get("prev_tw"):
        out += ["", "### Watchlist（前一交易日收盤）", *_tw_watch_table(d["prev_tw"]["watchlist"])]
    out += ["", "## 今日除權息（上市）"]
    ex = d.get("exdiv_today")
    if ex is None:
        out.append("- 資料未齊")
    elif not ex:
        out.append("- 無")
    else:
        for e in ex:
            cash = (
                f"現金股利 {fmt(e['cash'], 4).rstrip('0').rstrip('.')} 元"
                if e.get("cash")
                else (e.get("cash_text") or "")
            )
            stock = f"、配股率 {e['stock_ratio']}" if e.get("stock_ratio") else ""
            out.append(f"- {e['code']} {e['name']}（{e['kind']}）{cash}{stock}")
    out += ["", f"## 美股 Watchlist（{d['us_session_date']} 收盤）", *_us_watch_table(d["us_watchlist"], "盤後")]
    return out


def _body_us_pre(d):
    out = ["## 指數期貨"] + [_chg_line(r["label"], r) for r in d["futures"]]
    out += [_chg_line("VIX（最新）", r) for r in d["vix_prev"]]
    out += ["", "## 利率、匯率與商品（產出時最新報價）"] + [_chg_line(r["label"], r) for r in d["rates_fx"]]
    out += ["", "## 亞股收盤"] + [_chg_line(r["label"], r) for r in d["asia"]]
    out += ["", "## 今日台股", *_tw_summary_lines(d.get("tw_today"), "台股")]
    if d.get("tw_today"):
        out += ["", "### 台股 Watchlist", *_tw_watch_table(d["tw_today"]["watchlist"])]
    out += ["", "## 美股 Watchlist 盤前（收盤欄為前一交易日）", *_us_watch_table(d["us_watchlist_pre"], "盤前")]
    return out


def _body_us_post(d):
    out = ["## 指數"] + [_chg_line(r["label"], r) for r in d["indices"]]
    out += ["", "## 利率、匯率與商品（產出時最新報價）"] + [_chg_line(r["label"], r) for r in d["rates_fx"]]
    out += ["", "## 類股 ETF（依漲跌排序）"]
    out += [
        f"- {r['label']}（{r['key']}）：{fmt(r['last'])}，{fmt_signed(r['chg'])}（{fmt_pct(r['pct'])}）"
        for r in d["sectors"]
    ]
    out += ["", "## Watchlist（收盤與盤後）", *_us_watch_table(d["watchlist"], "盤後")]
    a = d.get("adr")
    out += ["", "## 台積電 ADR 對台股"]
    if a:
        out += [
            f"- {a['adr']}：{fmt(a['adr_close'])} 美元（{fmt_pct(a['adr_pct'])}），折合 {fmt(a['equiv_twd'])} 元（匯率 {fmt(a['usdtwd'], 3)}）",
            f"- 相對 {a['tw_code']} {a['tw_date']} 收盤 {fmt_price(a['tw_close'])} 元：溢價 {fmt_pct(a['premium_pct'])}",
        ]
    else:
        out.append("- 資料未齊")
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
        "tw-postmarket": lambda: _body_tw_post(d, (cfg or {}).get("tw", {})),
        "tw-premarket": lambda: _body_tw_pre(d),
        "us-premarket": lambda: _body_us_pre(d),
        "us-postmarket": lambda: _body_us_post(d),
    }[kind]()
    parts += body
    parts += ["", "## 解讀（非投資建議）"]
    if section:
        parts += [section["interpretation"], "", "觀察重點："] + [f"- {w}" for w in section["watch_points"]]
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
    parts += ["", "## 資料來源"] + [f"- {s['name']}：{s['url']}" for s in facts["sources"]]
    parts += ["", DISCLAIMER, ""]
    md = "\n".join(parts)
    summary_lines = [facts["title"], det_one]
    if section:
        summary_lines.append("一句話：" + section["one_liner"])
    if facts["gaps"]:
        summary_lines.append("資料未齊：" + "；".join(facts["gaps"])[:150])
    summary = "\n".join(summary_lines)[:580]
    return md, summary
