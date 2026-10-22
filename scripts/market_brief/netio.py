"""HTTP layer for market_brief (stdlib only).

Named netio, not http, so it never shadows the stdlib `http` package that urllib imports.

`Fetcher` records every request (url, status, bytes, error) so the report can list its sources, and takes an injectable
`transport(method, url, headers, body, timeout) -> (status, bytes)` so tests run offline from saved fixtures.
Errors carry the URL and status only; no request headers are ever logged (the OpenRouter call goes through llm.py,
which builds its own headers and never passes them here for logging).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0.0.0 Safari/537.36"
)
DEFAULT_HEADERS = {
    "User-Agent": BROWSER_UA,
    "Accept": "application/json, text/plain, application/xml, text/xml, */*",
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
}


class FetchError(RuntimeError):
    def __init__(self, url, message, status=None):
        super().__init__(f"{message} ({short_url(url)})")
        self.url = url
        self.status = status


def short_url(url, limit=160):
    return url if len(url) <= limit else url[: limit - 3] + "..."


def urllib_transport(method, url, headers, body, timeout):
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


# Minimum seconds between requests to one host. TWSE is known to block IPs that burst requests.
HOST_MIN_INTERVAL_S = {"www.twse.com.tw": 2.5, "www.tpex.org.tw": 1.5, "www.taifex.com.tw": 1.5}


class Fetcher:
    def __init__(
        self, transport=None, sleep=time.sleep, retries=1, retry_sleep_s=2.0, timeout=25, clock=time.monotonic
    ):
        self.transport = transport or urllib_transport
        self.sleep = sleep
        self.clock = clock
        self.retries = retries
        self.retry_sleep_s = retry_sleep_s
        self.timeout = timeout
        self.log = []  # [{url, status, bytes, error}]
        self._last_hit = {}

    def _throttle(self, url):
        host = urllib.parse.urlsplit(url).netloc
        gap = HOST_MIN_INTERVAL_S.get(host)
        if gap is None:
            return
        last = self._last_hit.get(host)
        if last is not None:
            wait = gap - (self.clock() - last)
            if wait > 0:
                self.sleep(wait)
        self._last_hit[host] = self.clock()

    def request(self, url, *, method="GET", params=None, body=None, headers=None, retries=None):
        if params:
            url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
        h = dict(DEFAULT_HEADERS)
        h.update(headers or {})
        tries = (self.retries if retries is None else retries) + 1
        last_err, last_status = None, None
        for attempt in range(tries):
            self._throttle(url)
            try:
                status, raw = self.transport(method, url, h, body, self.timeout)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err, last_status = f"network error {type(e).__name__}", None
            else:
                if status < 300:
                    self.log.append({"url": url, "status": status, "bytes": len(raw or b""), "error": None})
                    return raw
                last_err, last_status = f"HTTP {status}", status
                if status < 500 and status != 429:
                    break
                if status == 429:  # rate limited: do not hammer
                    break
            if attempt + 1 < tries:
                self.sleep(self.retry_sleep_s)
        self.log.append({"url": url, "status": last_status, "bytes": 0, "error": last_err})
        raise FetchError(url, last_err, last_status)

    def get_json(self, url, **kw):
        raw = self.request(url, **kw)
        try:
            return json.loads(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, ValueError):
            self.log[-1]["error"] = "non-JSON body"
            raise FetchError(url, "non-JSON body") from None

    def post_json(self, url, payload, **kw):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        headers.update(kw.pop("headers", None) or {})
        raw = self.request(url, method="POST", body=body, headers=headers, **kw)
        try:
            return json.loads(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, ValueError):
            raise FetchError(url, "non-JSON body") from None
