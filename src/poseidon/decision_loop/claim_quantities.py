"""Numeric/unit tokenizer and clause lexicon shared by the claim checks (one copy).

``poseidon.decision_loop.claim_lint`` (the server's completion gate) and ``poseidon.research_runner.claim_precheck``
(the research runner's gates) both build on these: the lexicons, ``Quantity``, the raw/word tokenizers, ``_annotate``
(qualifier, sign, growth period, table unit), the matching helpers and the atomic-lint split lexicon. What differs
between the two (the X ± Y merge, the derivation search, the inference rules) stays in each module; see their
docstrings. Stdlib only.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

TOL = 0.005  # relative tolerance for "same value"


# --------------------------------------------------------------------------------------------- lexicons
SCALES = [  # longest first
    ("百萬", 1e6),
    ("千萬", 1e7),
    ("千", 1e3),
    ("萬", 1e4),
    ("億", 1e8),
    ("兆", 1e12),
    ("thousand", 1e3),
    ("million", 1e6),
    ("billion", 1e9),
    ("trillion", 1e12),
    ("bn", 1e9),
    ("mn", 1e6),
]


CJK_SCALE = {"千": 1e3, "萬": 1e4, "百萬": 1e6, "千萬": 1e7, "億": 1e8, "兆": 1e12}


CUR_PREFIX = [
    ("NT$", "TWD"),
    ("US$", "USD"),
    ("JP¥", "JPY"),
    ("新台幣", "TWD"),
    ("新臺幣", "TWD"),
    ("台幣", "TWD"),
    ("臺幣", "TWD"),
    ("NTD", "TWD"),
    ("TWD", "TWD"),
    ("USD", "USD"),
    ("$", "USD"),
    ("¥", "JPY"),
    ("€", "EUR"),
]


CUR_SUFFIX = [
    ("美元", "USD"),
    ("日圓", "JPY"),
    ("日元", "JPY"),
    ("円", "JPY"),
    ("新台幣元", "TWD"),
    ("台幣", "TWD"),
    ("元", None),
    ("US dollars", "USD"),
    ("US dollar", "USD"),
    ("NT dollars", "TWD"),
    ("NT dollar", "TWD"),
    ("dollars", "USD"),
    ("dollar", "USD"),
    ("yen", "JPY"),
    ("USD", "USD"),
    ("NTD", "TWD"),
    ("TWD", "TWD"),
]


UNITS = [  # (token, class, multiplier); order matters (longest / most specific first)
    ("percentage points", "pp", 1.0),
    ("percentage point", "pp", 1.0),
    ("個百分點", "pp", 1.0),
    ("百分點", "pp", 1.0),
    ("basis points", "pp", 0.01),
    ("basis point", "pp", 0.01),
    ("個基點", "pp", 0.01),
    ("基點", "pp", 0.01),
    ("bps", "pp", 0.01),
    ("bp", "pp", 0.01),
    ("ppts", "pp", 1.0),
    ("ppt", "pp", 1.0),
    ("pp", "pp", 1.0),
    ("%", "percent", 1.0),
    ("％", "percent", 1.0),
    ("percent", "percent", 1.0),
    ("倍", "multiple", 1.0),
    ("times", "multiple", 1.0),
    ("-fold", "multiple", 1.0),
    ("fold", "multiple", 1.0),
    ("x", "multiple", 1.0),
    ("-nanometer", "nm", 1.0),
    ("nanometers", "nm", 1.0),
    ("nanometer", "nm", 1.0),
    ("奈米", "nm", 1.0),
    ("nm", "nm", 1.0),
    ("PB", "bytes", 1e15),
    ("TB", "bytes", 1e12),
    ("GB", "bytes", 1e9),
    ("MB", "bytes", 1e6),
    ("MT/s", "transfer_rate", 1.0),
    ("GT/s", "transfer_rate", 1.0),
    ("days", "days", 1.0),
    ("day", "days", 1.0),
    ("天", "days", 1.0),
]


ASCII_WORD_UNITS = {t for t, _, _ in UNITS if t[0].isascii() and t[0].isalpha()} | {
    "thousand",
    "million",
    "billion",
    "trillion",
    "bn",
    "mn",
}


_NUMBER_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty"
)


WORD_NUM = {w: i for i, w in enumerate(_NUMBER_WORDS.split())}


CJK_DIGIT = {"一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


POS_WORDS = (
    r"increased|increases|increase|rose|rise|rises|grew|grows|grow|gained|gain|up(?!\s+to\b)|jumped|surged|climbed|"
    r"年增|季增|月增|增加|增長|成長|上升|上漲|提升|增"
)


NEG_WORDS = (
    r"decreased|decreases|decrease|fell|falls|fall|declined|declines|decline|dropped|drop|down|shrank|contracted|"
    r"年減|季減|月減|減少|下降|下滑|下跌|衰退|減"
)


SIGN_RE = re.compile(rf"(?P<pos>{POS_WORDS})|(?P<neg>{NEG_WORDS})", re.I)


QUAL_PATTERNS = [  # checked against the text right before the number (and 以上/以下 right after)
    (">", r"(?:more than|over|above|exceeding|exceeded|surpassed|at least|in excess of|超過|逾|多於|高於)\s*$"),
    ("<~", r"(?:nearly|almost|close to|接近|將近|近)\s*$"),
    ("~", r"(?:approximately|approx\.?|about|around|roughly|some|約|大約|大概)\s*$"),
    ("<", r"(?:up to|less than|under|below|at most|不到|低於|未滿|最多)\s*$"),
]


GROWTH_TOKENS = [
    (
        "yoy",
        r"year[- ]over[- ]year|\byoy\b|\by/y\b|year[- ]ago|a year earlier|same (?:period|quarter) (?:last|a) year|"
        r"from (?:a|the prior) year|prior year|年增|年減|年成長|較去年同期|相較去年|與去年同期",
    ),
    (
        "qoq",
        r"quarter[- ]over[- ]quarter|\bqoq\b|\bq/q\b|sequential(?:ly)?|(?:prior|previous|preceding) quarter|"
        r"(?:from|compared (?:to|with)) (?:the )?(?:first|second|third|fourth) quarter|季增|季減|較上季|上季",
    ),
    ("mom", r"month[- ]over[- ]month|\bmom\b|\bm/m\b|(?:prior|previous) month|月增|月減|較上月"),
]


GROWTH_RE = re.compile("|".join(f"(?P<{k}>{p})" for k, p in GROWTH_TOKENS), re.I)


MONTH_BEFORE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
    r"Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s*$"
)


TABLE_UNIT_RE = re.compile(
    r"[\(（]\s*(?:Unit|Units|單位)\s*[:：]\s*(?P<cur>NT\$|US\$|新台幣|新臺幣|美元)?\s*"
    r"(?P<scale>thousands?|millions?|billions?|百萬|千|億)",
    re.I,
)


NUM_RE = re.compile(r"(?<![A-Za-z0-9.])(?P<sign>[+\-−])?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")


@dataclass
class Quantity:
    raw: str
    span: tuple
    value: float  # magnitude in base unit (currency unit, %, pp, x, nm, bytes, ...)
    unit_class: str
    currency: str | None = None
    period: str | None = None  # growth token yoy/qoq/mom
    sign: int = 0  # +1 / -1 / 0 (unknown)
    qualifier: str | None = None  # > < <~ ~
    range: tuple | None = None  # (lo, hi) in base unit
    pm: float | None = None  # half-width for X ± Y
    table: bool = False  # bare number under a (Unit: ...) table header
    table_money: tuple | None = None  # (value, currency) under the table's declared unit
    # parse bookkeeping (not part of the canonical form)
    scale: float = field(default=1.0, repr=False)
    scale_tok: str | None = field(default=None, repr=False)
    has_unit: bool = field(default=False, repr=False)

    def canonical(self):
        return (self.value, self.unit_class, self.currency, self.period)

    def to_dict(self):
        d = asdict(self)
        for k in ("scale", "scale_tok", "has_unit"):
            d.pop(k)
        return {k: v for k, v in d.items() if v not in (None, False) or k in ("value", "unit_class")}


# --------------------------------------------------------------------------------------------- helpers
def _close(a, b, tol=TOL):
    if a == b:
        return True
    return abs(a - b) <= tol * max(abs(a), abs(b))


def _take(text, pos, tokens, ascii_word_boundary=True):
    """Match one of `tokens` ((tok, *payload) tuples) at text[pos:], allowing one optional space before it."""
    for skip in (0, 1):
        p = pos + skip
        if skip and (p > len(text) or not text[pos:p].isspace()):
            continue
        for item in tokens:
            tok = item[0]
            if text.startswith(tok, p) or (tok.islower() and text[p : p + len(tok)].lower() == tok):
                end = p + len(tok)
                if (
                    ascii_word_boundary
                    and tok[-1].isascii()
                    and tok[-1].isalpha()
                    and end < len(text)
                    and text[end].isascii()
                    and text[end].isalnum()
                ):
                    continue
                if tok in ("x",) and skip:  # "7 x" is not a multiple
                    continue
                return item, end
    return None, pos


def _sentences(text):
    """Spans of sentence-ish segments (for period tokens)."""
    out, start = [], 0
    for m in re.finditer(r"[。；;!?！？•\n]|(?<!\d)\.(?=\s|$)|(?<=\D\d)\.(?=\s+[A-Z])", text):
        out.append((start, m.end()))
        start = m.end()
    out.append((start, len(text)))
    return out


def _clauses(text):
    out, start = [], 0
    for m in re.finditer(r"[。；;!?！？•\n,，、]|(?<!\d)\.(?=\s|$)", text):
        out.append((start, m.end()))
        start = m.end()
    out.append((start, len(text)))
    return out


def _seg(spans, pos):
    for s, e in spans:
        if s <= pos < e:
            return s, e
    return 0, 0


def _nearest(regex, text, seg, s, e, maxdist):
    best, bd = None, maxdist + 1
    for m in regex.finditer(text, seg[0], seg[1]):
        if m.end() <= s:
            d = s - m.end()
        elif m.start() >= e:
            d = m.start() - e
        else:
            continue
        if d < bd:
            best, bd = m, d
    return best


# --------------------------------------------------------------------------------------------- extraction
def _skip_spans(text):
    spans = []
    for pat in (
        r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",
        r"(?<![\d.,])\d{1,2}-\d{1,2}(?![\d.%])",
        r"(?<![\d.])\d+/\d+(?![\d.])",
    ):
        for m in re.finditer(pat, text):
            tail = text[m.end() : m.end() + 12]
            if pat.startswith(r"(?<![\d.,])") and re.match(
                r"\s*(?:%|％|percent|倍|times|億|萬|兆|billion|million)", tail
            ):
                continue  # 10-15% is a range, not a date
            spans.append(m.span())
    return spans


SIZE_RE = re.compile(r"\s*(?:-\s*)?(?:吋|英吋|inch(?:es)?\b|nm\b|奈米|mm\b|毫米|微米|μm\b|um\b)")


def _raw_tokens(text):
    toks = []
    skip = _skip_spans(text)
    for m in NUM_RE.finditer(text):
        s, e = m.span("num")
        if any(a <= s < b for a, b in skip):
            continue
        num = m.group("num")
        # footnote markers like net(2) / GAAP(1)
        if s >= 2 and text[s - 1] == "(" and text[e : e + 1] == ")" and text[s - 2].isalpha():
            continue
        # identifier tail after a hyphen: FQ1-27
        if s >= 2 and text[s - 1] == "-" and text[s - 2].isalnum():
            k = s - 2
            while k >= 0 and text[k].isascii() and text[k].isalnum():
                k -= 1
            if any(ch.isalpha() for ch in text[k + 1 : s - 1]):
                continue
        pre = text[max(0, s - 12) : s]
        pre_strip = pre.rstrip()
        if pre_strip.endswith("第") or MONTH_BEFORE.search(pre):
            continue
        value = float(num.replace(",", ""))
        pos = e
        if SIZE_RE.match(text[e : e + 12]):
            continue  # wafer size / process node / dimension descriptor: 8 吋, 12-inch, 28nm, 0.18 微米
        # scale
        item, pos2 = _take(text, pos, SCALES)
        scale, scale_tok = 1.0, None
        if item:
            scale_tok, scale = item
            pos = pos2
        # currency suffix
        currency, has_cur = None, False
        item, pos2 = _take(text, pos, CUR_SUFFIX)
        if item:
            currency, has_cur = item[1], True
            pos = pos2
        # unit
        unit_class, mult, has_unit = None, 1.0, False
        item, pos2 = _take(text, pos, UNITS)
        if item and not (item[0] == "x" and pos2 - pos != 1):
            _, unit_class, mult = item
            has_unit = True
            pos = pos2
        # glued identifiers: 3Q26, 20s, 1H, LPDDR6x, 1-gamma
        if not (has_unit or has_cur or scale_tok):
            nxt = text[e : e + 1]
            if nxt.isascii() and nxt.isalpha():
                if "." in num and re.match(r"[a-c](?![A-Za-z])", text[e : e + 2]):
                    pass  # table footnote letter: 29.43b
                else:
                    continue
            if nxt == "-" and re.match(r"-[A-Za-z]", text[e : e + 2]):
                continue
            if nxt == "/" and text[e + 1 : e + 2].isdigit():
                continue
        after = text[pos : pos + 4].lstrip()
        if (
            not (has_unit or has_cur)
            and after[:1] in ("年", "月", "日", "號", "季")
            and not re.match(r"年[增減成]", after)
        ):
            continue
        # currency prefix
        pre_cur, qstart = None, m.start()
        lead = text[max(0, m.start() - 8) : m.start()].rstrip()
        for tok, cur in CUR_PREFIX:
            if lead.endswith(tok):
                pre_cur = cur
                qstart = max(0, m.start() - 8) + len(lead) - len(tok)
                break
        if pre_cur:
            currency, has_cur = pre_cur, True
        if has_cur and currency is None and not unit_class:
            pass
        bare = not (has_unit or has_cur or scale_tok)
        if (
            bare
            and "." not in num
            and "," not in num
            and (1900 <= value <= 2100 or value >= 1000 or (len(num) > 1 and num[0] == "0"))
        ):
            continue  # year / ticker (2330, 0050) / product number
        if has_cur:
            cls = "money"
        elif unit_class:
            cls = unit_class
        else:
            cls = "count"
        q = Quantity(
            raw=text[qstart:pos],
            span=(qstart, pos),
            value=value * scale * mult,
            unit_class=cls,
            currency=currency,
            scale=scale,
            scale_tok=scale_tok,
            has_unit=has_unit or has_cur,
        )
        if m.group("sign") and (m.start() == 0 or not text[m.start() - 1].isalnum()):
            q.sign = 1 if m.group("sign") == "+" else -1
        # 以上 / 以下 after the quantity
        tail = text[pos : pos + 3]
        if tail.startswith("以上"):
            q.qualifier = ">"
        elif tail.startswith("以下"):
            q.qualifier = "<"
        toks.append(q)
    return toks


def _word_tokens(text):
    toks = []
    unit_alt = r"times|days?|percent|billion|million|thousand|trillion|percentage points?|basis points?"
    for m in re.finditer(rf"\b(?P<w>{'|'.join(WORD_NUM)})(?P<fold>fold)?\b(?=(?P<u>\s+(?:{unit_alt})\b)?)", text, re.I):
        if not (m.group("fold") or m.group("u")):
            continue
        v = float(WORD_NUM[m.group("w").lower()])
        if m.group("fold"):
            toks.append(Quantity(raw=m.group(0), span=m.span(), value=v, unit_class="multiple", has_unit=True))
            continue
        pos = m.end()
        item, pos2 = _take(text, pos, SCALES)
        scale = 1.0
        if item:
            scale = item[1]
            pos = pos2
        cls, mult, cur = "count", 1.0, None
        item, pos2 = _take(text, pos, CUR_SUFFIX)
        if item:
            cls, cur, pos = "money", item[1], pos2
        item, pos2 = _take(text, pos, UNITS)
        if item:
            cls, mult, pos = item[1], item[2], pos2
        toks.append(
            Quantity(
                raw=text[m.start() : pos],
                span=(m.start(), pos),
                value=v * scale * mult,
                unit_class=cls,
                currency=cur,
                scale=scale,
                has_unit=True,
            )
        )
    for m in re.finditer(r"\b(doubled|tripled|quadrupled)\b|翻倍", text, re.I):
        v = {"doubled": 2, "tripled": 3, "quadrupled": 4}.get((m.group(1) or "").lower(), 2)
        toks.append(Quantity(raw=m.group(0), span=m.span(), value=float(v), unit_class="multiple", has_unit=True))
    for m in re.finditer(r"([一二兩三四五六七八九十])倍", text):
        v = CJK_DIGIT[m.group(1)]
        if v == 1 and re.search(r"(?:增|成長|增加|增長)\D{0,4}$", text[max(0, m.start() - 8) : m.start()]):
            v = 2  # 增加一倍 / 年增超過一倍 == doubled
        toks.append(Quantity(raw=m.group(0), span=m.span(), value=float(v), unit_class="multiple", has_unit=True))
    return toks


def _annotate(text, qs):
    sents, clauses = _sentences(text), _clauses(text)
    tm = TABLE_UNIT_RE.search(text)
    for q in qs:
        q.value = float(f"{q.value:.12g}")  # drop float noise: 1,331.9 * 1e8
        if q.range:
            q.range = tuple(float(f"{v:.12g}") for v in q.range)
        s, e = q.span
        pre = text[max(0, s - 24) : s]
        if q.qualifier is None:
            for sym, pat in QUAL_PATTERNS:
                if re.search(pat, pre, re.I):
                    q.qualifier = sym
                    break
        if (
            q.sign == 0
            and q.unit_class not in ("nm", "bytes", "transfer_rate")
            and not re.search(r"(?:\bto|\bat|至|為|達|是)\s*$", pre, re.I)
        ):
            m = _nearest(SIGN_RE, text, _seg(clauses, s), s, e, 30)
            if m:
                q.sign = 1 if m.group("pos") else -1
        m = _nearest(GROWTH_RE, text, _seg(sents, s), s, e, 80)
        if m:
            q.period = next(k for k, v in m.groupdict().items() if v)
        if tm and s > tm.end() and not q.has_unit and q.unit_class == "count" and q.range is None:
            sc = tm.group("scale").lower().rstrip("s")
            mult = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "百萬": 1e6, "千": 1e3, "億": 1e8}.get(sc, 1.0)
            cur = {"NT$": "TWD", "US$": "USD", "新台幣": "TWD", "新臺幣": "TWD", "美元": "USD"}.get(
                tm.group("cur") or ""
            )
            q.table, q.table_money = True, (q.value * mult, cur)
    return qs


# --------------------------------------------------------------------------------------------- numeric check
def _points(q: Quantity):
    """(value, class, currency, kind, quantity) candidate points of an excerpt quantity."""
    pts = []
    if q.range:
        lo, hi = q.range
        pts += [
            (lo, q.unit_class, q.currency, "direct", q),
            (hi, q.unit_class, q.currency, "direct", q),
            ((lo + hi) / 2, q.unit_class, q.currency, "derived:range_midpoint", q),
        ]
    elif q.pm is not None:
        pts += [
            (q.value, q.unit_class, q.currency, "direct", q),
            (q.pm, q.unit_class, q.currency, "direct", q),
            (q.value - q.pm, q.unit_class, q.currency, "derived:minus_pm", q),
            (q.value + q.pm, q.unit_class, q.currency, "derived:plus_pm", q),
        ]
    else:
        pts.append((q.value, q.unit_class, q.currency, "direct", q))
    if q.table:
        pts.append((q.value, "*", None, "direct:table_bare", q))
        if q.table_money:
            pts.append((q.table_money[0], "money", q.table_money[1], "direct:table_unit", q))
    return pts


def _compatible(cls, cur, pcls, pcur):
    if pcls == "*":
        return True
    if cls != pcls:
        return False
    return not (cls == "money" and cur and pcur and cur != pcur)


def _bound_ok(qual, v, x):
    if qual == ">":
        return v <= x <= 1.5 * v or _close(v, x)
    if qual == "<~":
        return 0.9 * v <= x <= v or _close(v, x)
    if qual == "~":
        return _close(v, x, 0.05)
    if qual == "<":
        return 0 < x <= v or _close(v, x)
    return _close(v, x)


def _growth_like(q):
    return q.unit_class in ("percent", "pp", "multiple") or q.sign != 0


# --------------------------------------------------------------------------------------------- atomic lint
_EN_VERBS = (
    r"is|are|was|were|be|been|being|has|have|had|will|would|expects?|expected|anticipates?|projects?|projected|plans?|"
    r"planned|guided|guides|rose|rises?|risen|fell|falls?|grew|grows?|increased|decreased|declined|dropped|surged|jumped|"
    r"raised|cut|reported|announced|represented|accounted|reached|began|begins?|completed|signed|launched|received|"
    r"includes?|assumes?|gained|lost|exceeded|surpassed|doubled|tripled|shipping|ships?|shipped|build|built|approved|"
    r"contribute[sd]?|outperformed|underperformed|beat|missed|delivered|totals?|totaled|equals?|remains?|"
    r"remained|became|achieved|secured|sampling|sampled|said|says|stated|told|led|leads?|drove|drives|caused|causes"
)


_ZH_VERBS = (
    r"為|是|有|增|減|成長|上升|下降|下滑|上漲|下跌|提升|降低|高於|低於|超過|達|占|佔|預期|預計|預估|宣布|推出|取得|簽署|核准|"
    r"完成|開始|合作|支付|調漲|調降|恢復|導致|維持|擴大|縮小|貢獻|出貨|相當於|公布|採用|發表|收購|移轉|表示|指出|帶動|趨緩"
)


_PRED_RE = re.compile(rf"\b(?:{_EN_VERBS})\b|{_ZH_VERBS}", re.I)


_CAUSAL = (
    r"\bdriven\s+(?:mainly\s+|primarily\s+|largely\s+|mostly\s+)?by\b|\bdue\s+to\b|\bbecause(?:\s+of)?\b|\bas\s+a\s+result\s+of\b|"
    r"\bthanks\s+to\b|\bowing\s+to\b|\bled\s+by\b|是因為|因為|由於|主要受|受惠於|歸因於|源於|帶動"
)


_SPLIT_RE = re.compile(
    rf"(?P<causal>{_CAUSAL})|(?P<delim>[。；;!?！？,，、]|(?<!\d)\.(?=\s|$)|\b(?:and|while|but|whereas|although|however)\b|"
    r"並且|同時|以及|而且|但是|然而|(?<=[，,\s])[且但])",
    re.I,
)
