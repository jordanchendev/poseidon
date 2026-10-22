"""Headline evidence from RSS feeds and the cnyes news-list API (stdlib only).

Only titles, links, source names and publish times are kept; they are passed to the model as numbered evidence and the
model may only reference them as [n]. Links lose their query string/fragment (tracking parameters; also keeps anything
token-like out of the vault). Titles are plain text, whitespace-collapsed, capped.
"""

from __future__ import annotations

import datetime as dt
import email.utils
import re
import urllib.parse
import xml.etree.ElementTree as ET

from netio import FetchError
from numfmt import strip_html

TITLE_CAP = 160


def clean_title(s):
    t = strip_html(s)
    t = "".join(ch for ch in t if ch.isprintable())
    return re.sub(r"\s+", " ", t).strip()[:TITLE_CAP]


def clean_link(url):
    try:
        p = urllib.parse.urlsplit(str(url or "").strip())
    except ValueError:
        return None
    if p.scheme not in ("http", "https") or not p.netloc or "@" in p.netloc:
        return None
    return urllib.parse.urlunsplit((p.scheme, p.netloc, p.path, "", ""))


def parse_rss(raw, source):
    """RSS 2.0 / Atom bytes -> [{title, link, published (aware UTC datetime or None), source}]."""
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []
    items = []
    for it in root.iter():
        tag = it.tag.rsplit("}", 1)[-1]
        if tag not in ("item", "entry"):
            continue
        fields = {c.tag.rsplit("}", 1)[-1]: c for c in it}
        title = clean_title(fields["title"].text if "title" in fields else "")
        link_el = fields.get("link")
        link = None
        if link_el is not None:
            link = link_el.text or link_el.get("href")
        pub = None
        for k in ("pubDate", "published", "updated", "date"):
            if k in fields and fields[k].text:
                pub = _parse_time(fields[k].text)
                if pub:
                    break
        link = clean_link(link)
        if title and link:
            items.append({"title": title, "link": link, "published": pub, "source": source})
    return items


def _parse_time(s):
    s = s.strip()
    try:
        t = email.utils.parsedate_to_datetime(s)
    except (TypeError, ValueError):
        t = None
    if t is None:
        try:
            t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.UTC)
    return t.astimezone(dt.UTC)


def parse_cnyes(doc, source):
    out = []
    for x in ((doc.get("items") or {}).get("data")) or []:
        nid, title, ts = x.get("newsId"), clean_title(x.get("title")), x.get("publishAt")
        if not nid or not title:
            continue
        pub = dt.datetime.fromtimestamp(int(ts), dt.UTC) if isinstance(ts, (int, float)) else None
        out.append(
            {"title": title, "link": f"https://news.cnyes.com/news/id/{int(nid)}", "published": pub, "source": source}
        )
    return out


def select(items_by_feed, now_utc, lookback_hours, per_feed, max_items):
    """Recent (published within lookback, not in the future by >10 min), deduped by title, newest first,
    at most per_feed per feed, max_items overall. Items without a publish time are dropped."""
    lo = now_utc - dt.timedelta(hours=lookback_hours)
    hi = now_utc + dt.timedelta(minutes=10)
    seen, picked = set(), []
    for items in items_by_feed:
        fresh = sorted(
            (i for i in items if i["published"] and lo <= i["published"] <= hi),
            key=lambda i: i["published"],
            reverse=True,
        )
        n = 0
        for i in fresh:
            key = re.sub(r"\W+", "", i["title"]).lower()
            if key in seen:
                continue
            seen.add(key)
            picked.append(i)
            n += 1
            if n >= per_feed:
                break
    picked.sort(key=lambda i: i["published"], reverse=True)
    return picked[:max_items]


def fetch_news(fetcher, cfg_news, market, now_utc, log=None):
    """-> (headlines [{n, title, link, source, published ISO}], feed_status [{id, ok, items}])."""
    per_feed_items, status = [], []
    for feed in cfg_news["feeds"]:
        if market not in feed.get("markets", []):
            continue
        try:
            if feed["type"] == "cnyes":
                items = parse_cnyes(fetcher.get_json(feed["url"]), feed["name"])
            else:
                items = parse_rss(fetcher.request(feed["url"]), feed["name"])
        except FetchError as e:
            if log:
                log(f"news feed {feed['id']} failed: {e}")
            status.append({"id": feed["id"], "ok": False, "items": 0})
            continue
        status.append({"id": feed["id"], "ok": True, "items": len(items)})
        per_feed_items.append(items)
    picked = select(per_feed_items, now_utc, cfg_news["lookback_hours"], cfg_news["per_feed"], cfg_news["max_items"])
    out = []
    for n, i in enumerate(picked, 1):
        out.append(
            {
                "n": n,
                "title": i["title"],
                "link": i["link"],
                "source": i["source"],
                "published": i["published"].isoformat(timespec="minutes").replace("+00:00", "Z"),
            }
        )
    return out, status
