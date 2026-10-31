#!/usr/bin/env python3
"""Fetch a primary-source document into an isolated directory and emit a `documents[]` block for research_theme.py.

- Downloads the URL (SEC EDGAR gets the required identifying User-Agent; others a browser UA), refuses Cloudflare
  challenge pages, records sha256 of the raw file and of the text conversion (HTML -> text via stdlib parser,
  PDF -> text via `pdftotext -layout`).
- Proposes excerpts: paragraphs (blank-line separated, page footers removed as whole lines unless a paragraph runs
  on across one) that match template anchors for the document type; each proposal is verified to be a verbatim substring of the text
  file under the same canonicalisation research_theme.py uses, so what you keep will freeze.
- Prints JSON: {"document": {...}, "proposed_excerpts": [...]}; copy `document` into the config and prune the
  excerpt list by hand (or pass --keep all).

Usage: fetch_source.py URL --id fundamental:tsmc-3q26-6k --title "..." --publisher "..." --event-time 2026-10-15T10:40:56Z
       --out-dir sources/ [--template earnings|press_release|generic] [--keep all|N,N,N]
Run with python3 -I from outside the download directory (untrusted data).
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

UA_SEC = os.environ.get("SEC_USER_AGENT", "jordan.chen.dev@gmail.com research-tool")
UA_BROWSER = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"
)
TEMPLATES = {
    "earnings": [
        r"revenue|net sales|營收",
        r"gross (profit )?margin|毛利",
        r"EPS|earnings per share|每股",
        r"guidance|outlook|expect|預期|展望|指引",
        r"capital expenditure|capex|資本支出",
        r"demand|需求",
    ],
    "press_release": [
        r"announce|宣布|signed|簽訂|agreement|合作|聯盟|投資",
        r"revenue|營收|million|億",
        r"expect|預計|預期|將",
    ],
    "generic": [r"."],
}
FOOTER = re.compile(r"^\s*(©|\(c\)|Page \d+|\d+\s*/\s*\d+\s*$)", re.I)
# A paragraph whose last line ends otherwise runs on across a page footer (acceptance 2026-10-30, N18).
SENTENCE_END = (".", "。", "!", "?", "！", "？", ":", "：", '"', "'", "”", "’", "」", "』", ")", "）", "]", "】")


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out = []
        self.skip = False

    def handle_starttag(self, t, a):
        if t in ("script", "style"):
            self.skip = True
        if t in ("p", "div", "tr", "br", "li", "table", "h1", "h2", "h3", "h4"):
            self.out.append("\n")
        if t in ("td", "th"):
            self.out.append(" | ")

    def handle_endtag(self, t):
        if t in ("script", "style"):
            self.skip = False

    def handle_data(self, d):
        if not self.skip:
            self.out.append(d)


def canonical(text):
    return " ".join(tok for tok in text.replace("\f", " ").split() if tok != "|")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(url, dest):
    ua = UA_SEC if "sec.gov" in url else UA_BROWSER
    req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read()
        ctype = r.headers.get("Content-Type", "")
    head = body[:4096].decode("utf-8", "replace").lower()
    if "just a moment" in head or "cf-chl" in head or "attention required" in head:
        sys.exit("blocked by a Cloudflare challenge page: %s" % url)
    Path(dest).write_bytes(body)
    return ctype


def to_text(raw_path, ctype):
    raw = Path(raw_path)
    is_pdf = raw.read_bytes()[:5] == b"%PDF-" or "pdf" in ctype
    txt = raw.with_suffix(raw.suffix + ".txt")
    if is_pdf:
        subprocess.run(["pdftotext", "-layout", str(raw), str(txt)], check=True)
    else:
        p = _Text()
        p.feed(raw.read_text(encoding="utf-8", errors="replace"))
        t = re.sub(r"[ \t\xa0]+", " ", "".join(p.out))
        t = re.sub(r"\n\s*\n+", "\n\n", t)
        txt.write_text(t, encoding="utf-8")
    return txt


def running_header(lines):
    """Lines that head two or more pages: the text at a form feed (pdftotext starts each page with one) up to the
    first blank line."""
    seen, state = Counter(), None  # "feed": a form feed, no text yet; "text": inside the header
    for line in lines:
        text = line.replace("\f", "").strip()
        if "\f" in line:
            state = "feed"
        if state and text:
            seen[text] += 1
            state = "text"
        elif state == "text":
            state = None
    return {t for t, k in seen.items() if k >= 2}


def paragraphs(text_path):
    """(first line, last line, text) per paragraph. A paragraph cut by a page footer before it ends a sentence runs on
    across the footer and the next page's running header, which stay in its text so that it is still a verbatim span
    of the file; the blank lines between them do not end it."""
    lines = Path(text_path).read_text(encoding="utf-8", errors="replace").split("\n")
    header = running_header(lines)
    paras, cur, start = [], [], None
    across = False  # inside the page break of a paragraph that runs on
    bullet = re.compile(r"^\s*([•\-\*▪·]|\(?\d{1,2}[.)])\s*\S")
    for n, line in enumerate(lines, 1):
        if "\f" in line:
            line = line.replace("\f", "")
        if across:
            if not line.strip():
                continue
            if FOOTER.match(line) or line.strip() in header:
                cur.append(line)
                continue
            across = False  # the paragraph goes on with this line
        elif cur and not cur[-1].rstrip().endswith(SENTENCE_END) and FOOTER.match(line):
            across = True
            cur.append(line)
            continue
        elif cur and not line.strip() and not cur[-1].rstrip().endswith(SENTENCE_END):
            following = next((x for x in lines[n:] if x.strip()), "")
            if FOOTER.match(following):
                continue  # the blank lines before a footer that cuts the paragraph
        if FOOTER.match(line) or not line.strip() or re.fullmatch(r"[ |]+", line):
            # a blank line does not end a paragraph whose last line ends with ':' (lead-in before bullets) or is a bullet
            if cur and (cur[-1].rstrip().endswith((":", "：", ",", "，", ";", "；")) or bullet.match(cur[-1])):
                continue
            if cur:
                paras.append((start, n - 1, " ".join(x.strip() for x in cur)))
                cur, start = [], None
            continue
        if (
            cur
            and not bullet.match(line)
            and bullet.match(cur[-1])
            and not cur[-1].rstrip().endswith((":", "：", ";", "；", ",", "，"))
        ):
            # bullets group ended by a non-bullet line: close the group first
            paras.append((start, n - 1, " ".join(x.strip() for x in cur)))
            cur, start = [], None
        if start is None:
            start = n
        cur.append(line)
    if cur:
        paras.append((start, len(lines), " ".join(x.strip() for x in cur)))
    return paras


def propose(text_path, template, min_chars=80):
    """Paragraphs of the converted text matching the template anchors, each pre-verified as a verbatim substring."""
    canon_all = canonical(Path(text_path).read_text(encoding="utf-8", errors="replace"))
    anchors = [re.compile(p, re.I) for p in TEMPLATES[template]]
    proposals = []
    for s, e, text in paragraphs(text_path):
        if len(text) < min_chars:
            continue
        hits = [p.pattern for p in anchors if p.search(text)]
        if not hits:
            continue
        ok = canonical(text) in canon_all
        proposals.append({"locator": "lines %d-%d" % (s, e), "text": text, "anchors": hits, "verified": ok})
    return proposals


def build_document(url, *, doc_id, title, publisher, event_time, out_dir, template="earnings", keep=None, min_chars=80):
    """Fetch `url` into out_dir, convert to text and build a research_theme.py `documents[]` entry.

    keep: None (no excerpts), "all" (every verified proposal) or a list of 1-based proposal indexes.
    Returns (document, proposals, content_type). Raises SystemExit on a Cloudflare challenge page and lets
    urllib / subprocess errors propagate (callers that must not die catch them).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", url.rsplit("/", 1)[-1] or "document")[:80] or "document"
    raw = out / name
    ctype = fetch(url, raw)
    txt = to_text(raw, ctype)
    proposals = propose(txt, template, min_chars)
    kept = []
    if keep == "all":
        kept = [p for p in proposals if p["verified"]]
    elif keep:
        kept = [proposals[int(i) - 1] for i in keep if proposals[int(i) - 1]["verified"]]
    doc = {
        "id": doc_id,
        "kind": "fundamental",
        "title": title,
        "publisher": publisher,
        "source_uri": url,
        "event_time": event_time,
        "available_at": event_time,
        "raw_text_path": str(txt),
        "raw_files": {raw.name: sha(raw), txt.name + " (converted)": sha(txt)},
        "excerpts": [{"locator": p["locator"], "text": p["text"]} for p in kept],
    }
    return doc, proposals, ctype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--id", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--publisher", required=True)
    ap.add_argument("--event-time", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--template", default="earnings", choices=sorted(TEMPLATES))
    ap.add_argument("--keep", default=None, help="'all' or comma-separated proposal indexes to keep as excerpts")
    ap.add_argument("--min-chars", type=int, default=80)
    a = ap.parse_args()
    doc, proposals, ctype = build_document(
        a.url,
        doc_id=a.id,
        title=a.title,
        publisher=a.publisher,
        event_time=a.event_time,
        out_dir=a.out_dir,
        template=a.template,
        keep="all" if a.keep == "all" else (a.keep.split(",") if a.keep else None),
        min_chars=a.min_chars,
    )
    print(
        json.dumps(
            {
                "document": doc,
                "fetched_at": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "content_type": ctype,
                "proposed_excerpts": [
                    {
                        "n": n,
                        "locator": p["locator"],
                        "verified": p["verified"],
                        "anchors": p["anchors"],
                        "preview": p["text"][:160],
                    }
                    for n, p in enumerate(proposals, 1)
                ],
            },
            ensure_ascii=False,
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
