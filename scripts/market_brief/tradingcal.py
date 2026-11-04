"""Trading calendars and slot scheduling.

TW: weekends + TWSE holidaySchedule (cached per year under <state>/cache/tw-holidays-<year>.json). If neither cache nor
endpoint is available, weekdays are assumed open (logged); closed-market detection on MI_INDEX still catches the rest.
US: weekends + NYSE full-day holidays from config (hard-coded per year).

A slot is due when slot_time <= now < slot_time + window, the session date is a trading day in that market, and the
state has no final entry for (kind, date). Slot times are wall-clock in the slot's tz. The US slots use Asia/Taipei (fixed times all year, owner decision
2026-11-04); `session_tz` (America/New_York) says which date they report, so holidays and the session follow New York.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import tw
from netio import FetchError

KINDS = ("us-postmarket", "tw-premarket", "tw-postmarket", "us-premarket")
FINAL_STATUSES = ("queued", "delivered", "closed", "failed")


class Calendar:
    def __init__(self, tw_closed_by_year=None, nyse_holidays=None):
        self.tw_closed_by_year = dict(tw_closed_by_year or {})  # {year: set(iso) or None (unknown)}
        self.nyse = set()
        for year, days in (nyse_holidays or {}).items():
            if not str(year).startswith("_"):
                self.nyse.update(days)

    def tw_open(self, d):
        if d.weekday() >= 5:
            return False
        closed = self.tw_closed_by_year.get(d.year)
        return True if closed is None else d.isoformat() not in closed

    def us_open(self, d):
        return d.weekday() < 5 and d.isoformat() not in self.nyse

    def is_open(self, market, d):
        return self.tw_open(d) if market == "tw" else self.us_open(d)

    def prev_open(self, market, d, max_back=15):
        """Latest trading day strictly before d."""
        x = d
        for _ in range(max_back):
            x -= dt.timedelta(days=1)
            if self.is_open(market, x):
                return x
        return None

    def nyse_known(self, year):
        return any(h.startswith(f"{year}-") for h in self.nyse)


def load_tw_holidays(fetcher, cache_dir, year, log=None):
    """Closed-day set for `year` from cache, else the TWSE endpoint (then cached). None when unavailable."""
    cache = Path(cache_dir) / f"tw-holidays-{year}.json"
    try:
        doc = json.loads(cache.read_text(encoding="utf-8"))
        if doc.get("year") == year and isinstance(doc.get("closed"), list):
            return set(doc["closed"])
    except (OSError, ValueError):
        pass
    try:
        closed = tw.parse_holidays(fetcher.get_json(tw.holiday_url(year)))
    except (FetchError, tw.NotReady) as e:
        if log:
            log(f"TW holiday schedule {year} unavailable ({e}); assuming weekdays open")
        return None
    if not any(c.startswith(f"{year}-") for c in closed):
        if log:
            log(f"TW holiday schedule {year} has no rows for that year yet; not cached")
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps({"year": year, "closed": closed}, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, cache)
    return set(closed)


def session_zone(slot):
    """Zone whose calendar date names the slot's session: the US slots fire at fixed Taipei times all year, but the
    session they report is the New York one (05:15 Taipei = 16:15/17:15 New York of the previous Taipei day)."""
    return ZoneInfo(slot.get("session_tz") or slot["tz"])


def _fires_on(slot, local_day):
    hh, mm = (int(x) for x in slot["time"].split(":"))
    return dt.datetime(local_day.year, local_day.month, local_day.day, hh, mm, tzinfo=ZoneInfo(slot["tz"]))


def slot_datetime(slot, d):
    """Instant the slot fires for session date d (a date in session_zone)."""
    for k in (0, 1, -1):  # the firing day in the slot tz is d, or d + 1 (05:15 Taipei reports the previous day)
        sdt = _fires_on(slot, d + dt.timedelta(days=k))
        if sdt.astimezone(session_zone(slot)).date() == d:
            return sdt
    raise ValueError(f"slot {slot} never fires for session {d}")


def due_slots(now_utc, slots, calendar, state, window_minutes):
    """-> [(kind, session_date, slot_dt, minutes_since_slot, market_open)] due at now_utc (aware). Checks today and
    yesterday in each slot tz so a window that crosses local midnight still resolves; the session date is the date of
    the slot instant in the slot's session zone. Closed-market slots are returned with market_open False (the runner
    records them once as closed, so they are logged once)."""
    out = []
    for kind in KINDS:
        slot = slots[kind]
        local_today = now_utc.astimezone(ZoneInfo(slot["tz"])).date()
        for fire_day in (local_today, local_today - dt.timedelta(days=1)):
            sdt = _fires_on(slot, fire_day)
            age = (now_utc - sdt).total_seconds() / 60.0
            if not (0 <= age < window_minutes):
                continue
            d = sdt.astimezone(session_zone(slot)).date()
            entry = state.get(f"{kind}:{d.isoformat()}") or {}
            if entry.get("status") in FINAL_STATUSES:
                continue
            out.append((kind, d, sdt, age, calendar.is_open(slot["market"], d)))
    return out
