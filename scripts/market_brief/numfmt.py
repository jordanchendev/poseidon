"""Number parsing/formatting shared by parsers, renderer and the LLM number check."""

from __future__ import annotations

import re

_TAG = re.compile(r"<[^>]+>")
WEEKDAY_ZH = "一二三四五六日"


def strip_html(s):
    return _TAG.sub("", str(s or "")).replace("&nbsp;", " ").strip()


def num(s):
    """'1,234.5' / '+0.17' / '5.084%' / 'UNCH' -> float; '', '--', '---', 'N/A', None -> None."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    t = strip_html(s).replace(",", "").replace("%", "").replace("$", "").strip()
    if t.upper() == "UNCH":
        return 0.0
    if t in ("", "-", "--", "---", "----", "N/A", "NULL", "null", "X", "除權息", "除息", "除權"):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def html_sign(s):
    """TWSE 漲跌(+/-) cell -> +1 / -1 / 0; None for X (不比價)."""
    t = strip_html(s)
    if "+" in t:
        return 1
    if "-" in t:
        return -1
    if "X" in t.upper():
        return None
    return 0


def pct_from_change(close, chg):
    if close is None or chg is None:
        return None
    prev = close - chg
    if prev <= 0:
        return None
    return chg / prev * 100.0


def fmt(x, nd=2, comma=True):
    if x is None:
        return "—"
    s = f"{x:,.{nd}f}" if comma else f"{x:.{nd}f}"
    return s


def fmt_signed(x, nd=2):
    if x is None:
        return "—"
    if round(x, nd) == 0:
        return f"{0:.{nd}f}"
    return ("+" if x > 0 else "") + f"{x:,.{nd}f}"


def fmt_pct(x, nd=2):
    if x is None:
        return "—"
    return fmt_signed(x, nd) + "%"


def yi(x):
    """元 -> 億元 (float)."""
    return None if x is None else x / 1e8


def fmt_price(x):
    """TW stock price: drop trailing zeros but keep up to 2 decimals (2,420 / 320.5 / 86.6 / 10.04)."""
    if x is None:
        return "—"
    s = f"{x:,.2f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def fmt_signed_price(x):
    """Signed TW price change without trailing zeros: +15 / -0.1 / 0."""
    if x is None:
        return "—"
    if round(x, 2) == 0:
        return "0"
    return ("+" if x > 0 else "-") + fmt_price(abs(x))


def weekday_zh(d):
    return WEEKDAY_ZH[d.weekday()]


def roc_to_iso(s):
    """'115/10/21' or '1151021' or '115年10月21日' -> '2026-10-21'; None if unparsable."""
    t = strip_html(s)
    m = re.match(r"^(\d{2,3})[/年](\d{1,2})[/月](\d{1,2})日?$", t)
    if not m:
        m = re.match(r"^(\d{3})(\d{2})(\d{2})$", t)
    if not m:
        return None
    y, mo, d = int(m.group(1)) + 1911, int(m.group(2)), int(m.group(3))
    return f"{y:04d}-{mo:02d}-{d:02d}"


def iso_to_roc_slash(iso):
    y, m, d = iso.split("-")
    return f"{int(y) - 1911}/{m}/{d}"
