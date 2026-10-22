#!/usr/bin/env python3
"""Read-only market brief server for Kairos (macminim4, Python 3.9 stdlib only).

Serves ONLY the Nexus vault folder _kb/raw/market-daily (the bridge writes every brief there):
  GET /index.json      -> {"generated_at", "briefs": [{date, kind, title, file, summary, bytes, session_date,
                          data_gaps}]} newest first, built from file names + frontmatter on every request
  GET /<name>.md       -> that file (text/markdown; charset=utf-8); name must match ^\\d{4}-\\d{2}-\\d{2}-[a-z-]+\\.md$
  anything else        -> 404; methods other than GET/HEAD -> 405; clients outside the allowlist -> 403
No directory listing, no paths, no symlinks, no other files. Public market data: no auth beyond the allowlist.

Install: ~/.openclaw/bin/brief_server.py + ~/Library/LaunchAgents/ai.aquarium.brief-server.plist (see that file).
Usage: brief_server.py [--port 8790] [--bind 0.0.0.0] [--root DIR] [--allow IP ...]
"""

import argparse
import datetime
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

DEFAULT_ROOT = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Nexus/_kb/raw/market-daily")
# stormtrooper LAN (also the source address of the kairos-proxy container, NATed via eno1), stormtrooper Tailscale,
# and loopback for local checks.
DEFAULT_ALLOW = ("192.168.31.181", "100.94.108.87", "127.0.0.1", "::1")
NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-([a-z-]+)\.md$")
LEGACY_KINDS = {"tw-market": "tw-postmarket-legacy"}
MAX_BYTES = 2_000_000


def parse_frontmatter(text):
    """Minimal YAML subset used by Nexus raw notes: `k: v` (JSON double-quoted, single-quoted or bare), `k: []`,
    `k:` followed by `- item` / `  - item` lines. Returns (dict, body)."""
    m = re.match(r"---\n(.*?)\n---\n?(.*)", text, re.S)
    if not m:
        return {}, text
    fm, key = {}, None
    for line in m.group(1).splitlines():
        s = line.strip()
        if s.startswith("- ") and key is not None:
            fm.setdefault(key, [])
            if isinstance(fm[key], list):
                fm[key].append(_scalar(s[2:]))
            continue
        if line[:1] in (" ", "\t") or ":" not in line:
            continue
        k, _, v = line.partition(":")
        key, v = k.strip(), v.strip()
        if v == "" or v == "[]":
            fm[key] = []
        elif v.startswith("[") and v.endswith("]"):
            fm[key] = [_scalar(x) for x in v[1:-1].split(",") if x.strip()]
        else:
            fm[key] = _scalar(v)
    return fm, m.group(2)


def _scalar(v):
    v = v.strip()
    if v.startswith('"'):
        try:
            return json.loads(v)
        except ValueError:
            return v.strip('"')
    if v.startswith("'") and v.endswith("'") and len(v) >= 2:
        return v[1:-1].replace("''", "'")
    if v in ("true", "false"):
        return v == "true"
    return v


def first_line_of(body, heading="一句話"):
    m = re.search(rf"^#{{1,6}}\s+{re.escape(heading)}\s*$\n+(.+)$", body, re.M)
    return m.group(1).strip() if m else ""


def build_index(root):
    briefs = []
    try:
        names = os.listdir(root)
    except OSError:
        names = []
    for name in names:
        m = NAME_RE.match(name)
        path = os.path.join(root, name)
        if not m or os.path.islink(path) or not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read(MAX_BYTES)
        except (OSError, UnicodeDecodeError):
            continue
        fm, body = parse_frontmatter(text)
        kind = LEGACY_KINDS.get(m.group(2), m.group(2))
        gaps = fm.get("data_gaps")
        briefs.append(
            {
                "date": m.group(1),
                "kind": kind,
                "title": str(fm.get("title") or name[:-3]),
                "file": name,
                "summary": first_line_of(body),
                "bytes": os.path.getsize(path),
                "session_date": str(fm.get("session_date") or m.group(1)),
                "data_gaps": gaps if isinstance(gaps, list) else [],
            }
        )
    briefs.sort(key=lambda b: (b["date"], not b["kind"].endswith("-legacy"), b["kind"]), reverse=True)
    return {"generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "briefs": briefs}


class Handler(BaseHTTPRequestHandler):
    server_version = "brief-server"
    sys_version = ""
    root = DEFAULT_ROOT
    allow = frozenset(DEFAULT_ALLOW)

    def _client_ok(self):
        ip = self.client_address[0]
        if ip.startswith("::ffff:"):
            ip = ip[7:]
        return ip in self.allow

    def _send(self, code, body=b"", ctype="text/plain; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _route(self):
        if not self._client_ok():
            return self._send(403, b"forbidden\n")
        path = urlsplit(self.path).path
        if path == "/index.json":
            body = json.dumps(build_index(self.root), ensure_ascii=False, indent=1).encode("utf-8")
            return self._send(200, body, "application/json; charset=utf-8")
        name = unquote(path[1:]) if path.startswith("/") else ""
        if "/" in name or "\\" in name or not NAME_RE.match(name):
            return self._send(404, b"not found\n")
        full = os.path.join(self.root, name)
        if os.path.islink(full) or not os.path.isfile(full):
            return self._send(404, b"not found\n")
        try:
            with open(full, "rb") as fh:
                data = fh.read(MAX_BYTES + 1)
        except OSError:
            return self._send(404, b"not found\n")
        if len(data) > MAX_BYTES:
            return self._send(404, b"not found\n")
        return self._send(200, data, "text/markdown; charset=utf-8")

    def do_GET(self):
        self._route()

    def do_HEAD(self):
        self._route()

    def _not_allowed(self):
        self._send(405, b"method not allowed\n")

    do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _not_allowed

    def log_message(self, fmt, *args):
        sys.stderr.write(
            "{} {} {}\n".format(
                datetime.datetime.now().isoformat(timespec="seconds"), self.client_address[0], fmt % args
            )
        )


def make_server(root, allow, bind="0.0.0.0", port=8790):
    handler = type("BriefHandler", (Handler,), {"root": root, "allow": frozenset(allow)})
    return ThreadingHTTPServer((bind, port), handler)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read-only market brief server (Nexus _kb/raw/market-daily).")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--allow", nargs="*", default=list(DEFAULT_ALLOW))
    a = ap.parse_args(argv)
    if not os.path.isdir(a.root):
        sys.stderr.write(f"root {a.root} not found\n")
    srv = make_server(a.root, a.allow, a.bind, a.port)
    sys.stderr.write(f"brief-server on {a.bind}:{a.port} root={a.root} allow={','.join(a.allow)}\n")
    srv.serve_forever()


if __name__ == "__main__":
    main()
