"""Phase 101 claim cadence, decision TTL, reconciliation age and OHLCV staleness.

A1 correction (D-23.14): TW 1d bars are stamped 05:30Z (13:30 Taipei close),
not 00:00Z/16:00Z. Authoritative timing table (all UTC; beat ``timezone="UTC"``):

| Event | UTC | Taipei | Notes |
|---|---|---|---|
| TW 1d bar time | 05:30 | 13:30 | bar stamp |
| Thalassa `fetch-tw-daily` | 06:00 | 14:00 | ingest (external) |
| legacy `evaluate-strategies-tw-daily` | 06:10 | 14:10 | legacy signals for released versions only |
| `decision-loop-reconcile-paper-account` | 06:20 daily (incl. Sat/Sun; `crontab(hour=6, minute=20)`, `celery_app.py:138`) | 14:20 | runs in shadow/decision/halted (101-03 Task 1) |
| `decision-loop-produce-tw-daily` (cutoff) | 06:40 | 14:40 | same-day bar age = 4,200 s |
| `decision-loop-freeze-outcome-manifests` (101-08) | 07:20 | 15:20 | |
| `decision-loop-label-outcomes` | 07:30 | 15:30 | |
| soak report host cron (101-07) | 08:00 | 16:00 | |
| operator review | before 01:00 next day | before 09:00 | D-18 |
| `portfolio-monthly-rebalance` claim | 01:30 Mon-Fri | 09:30 | this plan |
| decision `valid_until` = 06:40Z + 68,700 s | 01:45 next day | 09:45 | D-04 |
| `decision-loop-expire-decisions` | hourly :05 (02:05) | 10:05 | expires pending and approved |

``required_data.ohlcv.max_age_seconds`` = 14,400 s accepts the same-day 05:30Z
bar (up to 10,200 s of producer delay) and excludes every earlier bar
(weekends, holidays, ingest outages) as ``ohlcv_stale``.

Every schedule time below is derived from ``celery_app.conf.beat_schedule``;
only the policy values (TTL, max age, reconciliation minimum) are constants.
"""

from datetime import UTC, date, datetime, time, timedelta

import pytest

from poseidon.decision_loop.decisions import DecisionPolicy
from poseidon.decision_loop.producer import _evidence_reasons
from poseidon.workers.celery_app import celery_app
from tests.test_decision_service import synthetic_policy

TTL_SECONDS = 68_700
MAX_AGE_SECONDS = 14_400
# Reconcile 06:20Z on D-1 -> weekday claim 01:30Z on D = 69,000 s, also on Monday
# because the reconcile entry runs every day. Plus 600 s for reconcile run time
# and beat jitter. 101-06's policy validator enforces the same minimum; the owner
# chose this strict value (one missed 06:20Z reconcile fails the next claim closed;
# tolerating one miss would need >= 155,400 s).
MIN_RECONCILIATION_AGE_SECONDS = 69_600
RECONCILE_RUN_MARGIN_SECONDS = 600
BAR_TIME = time(5, 30)  # 05:30Z TW 1d bar stamp
PRODUCER = time(6, 40)

# Fixed reference week: Mon 2026-10-19 .. Mon 2026-10-26.
REFERENCE_MONDAY = date(2026, 10, 19)
REFERENCE_DAYS = [REFERENCE_MONDAY + timedelta(days=offset) for offset in range(8)]
WEEKDAYS = [day for day in REFERENCE_DAYS if day.weekday() < 5]
MON_TO_THU = [day for day in WEEKDAYS if day.weekday() < 4]
FRIDAY = date(2026, 10, 23)


def _entry(name):
    return celery_app.conf.beat_schedule[name]


def _fires_at(schedule, moment):
    celery_dow = (moment.weekday() + 1) % 7  # celery: Sunday == 0
    return (
        moment.minute in schedule.minute
        and moment.hour in schedule.hour
        and celery_dow in schedule.day_of_week
        and moment.day in schedule.day_of_month
        and moment.month in schedule.month_of_year
    )


def _next_fire(schedule, after, *, inclusive=False):
    moment = after.replace(second=0, microsecond=0)
    if not inclusive or moment < after:
        moment += timedelta(minutes=1)
    for _ in range(8 * 24 * 60):
        if _fires_at(schedule, moment):
            return moment
        moment += timedelta(minutes=1)
    raise AssertionError(f"no fire within 8 days after {after}")


def _previous_fire(schedule, before):
    moment = before.replace(second=0, microsecond=0)
    if moment >= before:
        moment -= timedelta(minutes=1)
    for _ in range(8 * 24 * 60):
        if _fires_at(schedule, moment):
            return moment
        moment -= timedelta(minutes=1)
    raise AssertionError(f"no fire within 8 days before {before}")


def _at(day, clock):
    return datetime.combine(day, clock, tzinfo=UTC)


def _producer_time():
    schedule = _entry("decision-loop-produce-tw-daily")["schedule"]
    (hour,) = schedule.hour
    (minute,) = schedule.minute
    return time(hour, minute)


def _claim_schedule():
    return _entry("portfolio-monthly-rebalance")["schedule"]


def _claim_slots():
    schedule = _claim_schedule()
    slots = [_next_fire(schedule, _at(day, time(0, 0)), inclusive=True) for day in WEEKDAYS]
    assert [slot.date() for slot in slots] == WEEKDAYS
    return slots


def _policy():
    return DecisionPolicy.model_validate(
        synthetic_policy(
            decision_ttl_seconds=TTL_SECONDS,
            required_data={"ohlcv": {"max_age_seconds": MAX_AGE_SECONDS}},
        )
    )


def test_producer_entry_is_unchanged_and_matches_cutoff_constant():
    entry = _entry("decision-loop-produce-tw-daily")
    assert entry["task"] == "poseidon.workers.cpu_tasks.produce_decision_runs_for_market"
    assert entry["kwargs"] == {"market": "tw_stock", "interval": "1d"}
    assert entry["schedule"].day_of_week == set(range(7))
    assert _producer_time() == PRODUCER
    assert (_at(REFERENCE_MONDAY, PRODUCER) - _at(REFERENCE_MONDAY, BAR_TIME)).total_seconds() == 4_200


@pytest.mark.parametrize("day", MON_TO_THU, ids=lambda day: day.strftime("%a-%m%d"))
def test_mon_to_thu_decision_is_claimable_next_weekday_then_swept(day):
    decision_as_of = _at(day, _producer_time())
    valid_until = decision_as_of + timedelta(seconds=TTL_SECONDS)
    next_claim = _next_fire(_claim_schedule(), decision_as_of)
    sweep = _next_fire(_entry("decision-loop-expire-decisions")["schedule"], valid_until, inclusive=True)
    next_day = day + timedelta(days=1)

    assert valid_until == _at(next_day, time(1, 45))
    assert next_claim == _at(next_day, time(1, 30))
    assert sweep == _at(next_day, time(2, 5))
    assert next_claim < valid_until < sweep


def test_friday_decision_expires_unclaimed_by_design():
    decision_as_of = _at(FRIDAY, _producer_time())
    valid_until = decision_as_of + timedelta(seconds=TTL_SECONDS)
    next_claim = _next_fire(_claim_schedule(), decision_as_of)
    sweep = _next_fire(_entry("decision-loop-expire-decisions")["schedule"], valid_until, inclusive=True)

    assert valid_until == _at(date(2026, 10, 24), time(1, 45))
    assert next_claim == _at(date(2026, 10, 26), time(1, 30))
    assert valid_until < sweep < next_claim


def test_reconcile_entry_runs_every_day_at_0620():
    entry = _entry("decision-loop-reconcile-paper-account")
    schedule = entry["schedule"]
    assert entry["task"] == "poseidon.workers.cpu_tasks.scheduled_reconcile_paper_account"
    assert schedule.hour == {6}
    assert schedule.minute == {20}
    assert schedule.day_of_week == set(range(7))


@pytest.mark.parametrize("slot_index", range(len(WEEKDAYS)), ids=lambda index: WEEKDAYS[index].strftime("%a-%m%d"))
def test_reconciliation_age_at_every_weekday_claim_is_69000_seconds(slot_index):
    claim = _claim_slots()[slot_index]
    reconcile_schedule = _entry("decision-loop-reconcile-paper-account")["schedule"]
    latest = _previous_fire(reconcile_schedule, claim)
    age = (claim - latest).total_seconds()

    assert age == 69_000
    assert age <= MIN_RECONCILIATION_AGE_SECONDS - RECONCILE_RUN_MARGIN_SECONDS

    # Owner note: one missed 06:20Z reconcile leaves the previous run 155,400 s old,
    # which the strict minimum rejects, so the claim fails closed.
    missed_one = _previous_fire(reconcile_schedule, latest)
    assert (claim - missed_one).total_seconds() == 155_400 > MIN_RECONCILIATION_AGE_SECONDS


def _ohlcv_reasons(bar_moment, cutoff):
    member = {"symbol": "2330"}
    evidence = [
        {
            "kind": "ohlcv",
            "event_time": bar_moment.isoformat().replace("+00:00", "Z"),
            "payload": {"symbol": "2330"},
        }
    ]
    return _evidence_reasons(member, evidence, _policy(), cutoff)


def test_same_day_0530_bar_is_fresh_at_the_0640_cutoff():
    day = REFERENCE_MONDAY + timedelta(days=1)
    assert _ohlcv_reasons(_at(day, BAR_TIME), _at(day, _producer_time())) == []


def test_bar_ingest_delay_up_to_10200_seconds_is_accepted():
    day = REFERENCE_MONDAY + timedelta(days=1)
    bar = _at(day, BAR_TIME)
    assert _ohlcv_reasons(bar, bar + timedelta(seconds=MAX_AGE_SECONDS)) == []
    assert _ohlcv_reasons(bar, bar + timedelta(seconds=MAX_AGE_SECONDS + 1)) == ["ohlcv_stale"]


def test_previous_session_0530_bar_is_stale():
    day = REFERENCE_MONDAY + timedelta(days=2)
    previous = day - timedelta(days=1)
    assert _ohlcv_reasons(_at(previous, BAR_TIME), _at(day, _producer_time())) == ["ohlcv_stale"]


@pytest.mark.parametrize(
    ("cutoff_day", "age_seconds"),
    [(date(2026, 10, 24), 90_600), (date(2026, 10, 25), 177_000), (date(2026, 10, 26), 263_400)],
    ids=["Sat", "Sun", "Mon"],
)
def test_friday_bar_is_stale_on_weekend_and_monday_cutoffs(cutoff_day, age_seconds):
    friday_bar = _at(FRIDAY, BAR_TIME)
    cutoff = _at(cutoff_day, _producer_time())
    assert (cutoff - friday_bar).total_seconds() == age_seconds
    assert _ohlcv_reasons(friday_bar, cutoff) == ["ohlcv_stale"]
