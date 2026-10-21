# Research loop stopgap tooling (2026-10-20)

> Moved 2026-10-21 from aquarium `local_dev/scripts/research/` into Poseidon `scripts/research/` (versioned). Paths below
> that start with `local_dev/scripts/research/` now mean `scripts/research/` in this repo; tests run from the repo root:
> `PYTHONPATH=src python -m pytest -q -p no:cacheprovider scripts/research/`. The server-side runner (`runner.py`) is
> described in RUNNER.md.

SEED-002 phase 2 is not implemented in Poseidon yet. These scripts close the loop end to end without
touching Poseidon production, so the loop can run daily while v21 is planned.

```
macminim4 (OpenClaw 2026.10.5)                      MacBook (this repo)
──────────────────────────────                      ────────────────────────────────────────────
cron 08:30 Asia/Taipei (weekdays)
  ~/.openclaw/bin/research-scout.sh                 agenda_pull.sh          -> accepted topics (JSON)
    agent "research-scout" (gpt-5.6-terra,          research_theme.py       -> freeze manifest, prepare,
      tools: web_search/web_fetch only)                                        complete, replay check
    -> dedupe vs ~/.openclaw/research/agenda.jsonl  render_theme.py         -> Markdown body
    -> verify primary/mirror URL fetchable          project_vault.py        -> Nexus Research/<scope>/<rev>.md
    -> append proposals (status=proposed)           agenda_done.sh          -> status=done + revision_id
    -> Telegram (main_bot DM): numbered list
Telegram reply "接受 N" / "略過 N" / "研究清單"
  -> main agent runs ~/.openclaw/bin/research-agenda.sh (rule in its AGENTS.md)
```

## Files on macminim4

- `~/.openclaw/bin/research_scout.py` + `research-scout.sh` — discovery run (cron job `research-scout-daily`, id 55fe47e7…).
- `~/.openclaw/bin/research_agenda.py` + `research-agenda.sh` — `list | pending | accept <N|id> | skip <N|id> | done <id>`.
- `~/.openclaw/research/agenda.jsonl` — canonical agenda (one JSON row per topic); `runs/<run_id>.json` raw agent output; `scout.log`.
- `~/.openclaw/workspace-research-scout/AGENTS.md` — scout rules and output schema.

## Fetch a primary source (MacBook)

```bash
python3 -I local_dev/scripts/research/fetch_source.py <url> --id fundamental:<slug> --title "..." --publisher "..." \
  --event-time 2026-10-15T10:40:56Z --out-dir <evidence-dir>/sources --template earnings|press_release|generic [--keep all|1,3,7]
```

Downloads into an isolated dir (SEC gets the identifying User-Agent; Cloudflare challenge pages are refused),
converts HTML/PDF to text, records both sha256s, and proposes paragraph excerpts matching the template anchors.
Every proposal is pre-verified as a verbatim substring (same canonicalisation as research_theme.py). Copy the
`document` object into the config `documents[]`; prune excerpts by hand or with `--keep`.

## Research run (MacBook)

Poseidon tests run only on stormtrooper, but the research contract (`ManifestService`, `ResearchService`)
imports cleanly in the local venv, so this runs against the deployed worktree:

```bash
cd poseidon/.worktrees/phase101-shadow-fix
PYTHONPATH=src ../../.venv/bin/python ../../../local_dev/scripts/research/research_theme.py \
  --config <config.json> --csv <thalassa bars csv> --ledger <ledger.sqlite> \
  [--narrative <narrative.json> --out <revision.json>]
```

- Config `documents[].excerpts[].text` must be a verbatim substring of `documents[].raw_text_path`
  (canonicalised: whitespace collapsed, bare `|` dropped). Cutting out page footers inside an excerpt
  fails verification by design: split into two excerpts instead.
- `symbol` in the config scopes the manifest to one instrument; the predecessor chain resolves on
  (account_scope, market, interval, symbol) + earlier `as_of`.
- The ledger is local SQLite; nothing is written to Poseidon. Completed revisions carry
  `origin: agent` in the vault until Poseidon owns projections (v21).

## Market reaction, catalysts, layered rendering (2026-10-21; gaps A4/A6/A7)

- `price-facts-v2` adds `ret_t1_pct`, `ret_t5_pct` (close of the 1st / 5th bar with time > `event_time` vs the last close
  <= `event_time`; null if not enough bars) and `rel_ret_t1_pct`, `rel_ret_t5_pct`, `rel_ret_since_event_pct` (symbol minus
  benchmark, percentage points). Benchmark = config `benchmark` (default `0050`) and must be listed in config `symbols`
  so its ohlcv is frozen in the manifest; otherwise relative fields are null. v1 fields are unchanged; full semantics in
  the `research_theme.py` docstring. Revisions frozen under v1 stay valid and render (new rows only appear when present).
- Optional narrative fields (extra top-level keys; Poseidon's `validate_research` ignores them):
  `catalysts: [{date YYYY-MM-DD, type monthly_revenue|earnings_call|agm|investor_day|product_event|regulatory|other, expected_signal, source}]`
  and `tl_dr` (one sentence, inference). `research_theme.py` validates the shape before completion and, when catalysts
  exist, requires `next_review_at` to start with the earliest catalyst date >= the as_of date (fails otherwise; if every
  catalyst is already past, the cross-check is skipped with a stderr note).
- `render_theme.py`: `tl_dr` blockquote labelled 推論 first; "催化劑日曆" table; fact claims as bullets under
  "事實（引用凍結證據）", inference under "推論", each bullet ending with its citations; T+1/T+5 absolute and relative rows in the price table.
- Tests (SQLite, no network): `cd poseidon/.worktrees/phase101-shadow-fix && PYTHONPATH=src ../../.venv/bin/python -m pytest -q -p no:cacheprovider ../../../local_dev/scripts/research/test_offline_tooling.py`

## Guidance bridge, KPI series, read-across, disclosure (2026-10-21; brief items B2/B3/B5/C1)

Design-note numbering differs: read-across is B4 there (B5 is variant perception). All fields below are optional; a
config without them freezes to the same manifest hash as before.

- **Guidance** — `documents[].guidance[]`: `{metric, period, low+high | value, unit, basis GAAP|non-GAAP|IFRS|n/a,
  fx_assumption?, excerpt_locator}`. Units: `%`, `pp`, `bp`, `x`, `USD|NTD|TWD|JPY [thousand|million|billion|trillion]`,
  anything else is a count. Every number (and the fx string) goes through `claim_precheck.check_numbers` against the excerpt
  with that locator: scale/sign mismatch refuses the freeze, value/absent only warns on stderr. A negative value also
  needs a negative/decrease counterpart. Guidance is stored in the document payload (hashed).
- **Actuals** — config `actuals[]: {metric, period, value, unit, evidence_id, excerpt_locator}` (any mismatch with the
  excerpt refuses; stored in that document's payload) or `claims[].actual {metric, period, value, unit}` on a fact claim
  (the number must be in the claim text). Reported period = `reported_period` or the actual periods.
- **Bridge** — the nearest completed predecessor in the same ledger (walking `previous_revision_id`) whose frozen
  documents carry guidance for the reported period gives `identity.guidance_bridge`; rendered as
  metric | prior guidance | actual | actual vs prior (above/within/below, same unit string only) | new guidance.
  No predecessor guidance -> only the current "本期指引" table.
- **KPI series** — `--kpi-json <file>` (list of raw `t187ap05_L` rows, or `{rows, source_uri, fetched_at}` for accumulated
  snapshots) or config `kpi_series: true` (live fetch of https://openapi.twse.com.tw/v1/opendata/t187ap05_L).
  Evidence `kpi_series:<symbol>` for `kpi_symbols` (default `symbols`; symbols without rows, e.g. ETFs, are skipped with a
  note): unit NTD thousand, last 12 months, a row counts only if its 出表日期 (end of that Taipei day) <= as_of,
  `trailing_3m_yoy_pct` = sum(last 3 months) / sum(their 去年當月營收) - 1 when the 3 months are consecutive.
  The live feed holds only the latest month per company, so history needs accumulated snapshots.
  Claims citing it are prechecked against the points rendered as text.
- **Read-across** — `read_across[]: {symbol, relation, why}`; T+1/T+5 relative returns come from price_facts when the
  symbol is in `symbols`, else n/a.
- **Precheck split** — fact claims: any numeric mismatch refuses. Inference claims: only scale/sign refuse; other
  mismatches are recorded as `numbers: mismatch-recorded`.
- **Disclosure** — last rendered section: origin, provider/model/policy, each evidence with its tier
  (`documents[].source_tier` overrides; otherwise sec.gov/MOPS -> primary filing, other documents -> company release,
  ohlcv/kpi_series -> exchange feed, price_facts -> derived), excerpts verified n/n, precheck counts, JEV from a
  `jev_results` block in the revision JSON (top level or research), and the not-investment-advice line.
- The price-table labels are derived from the price_facts `event_time` and the last bar; the unknowns header is "未知".

## Consensus & surprise, variant perception, pilot position (2026-10-21; design-note B1/B5/C1)

All optional; a run without them freezes to the same manifest hash as before.

- **Consensus JSON** (`--consensus-json <file>`, repeatable; manual/offline):
  `{symbol, provider, as_of (when the estimate was observed), source_uri, raw_path, raw_sha256, fetched_at, instrument?, note?,
  items: [{metric, period, mean, median?, low, high, n_analysts >= 1, unit, currency, basis GAAP|non-GAAP|IFRS|n/a}]}`.
  `raw_path` is the captured page/file and must hash to `raw_sha256` (refuses otherwise). Periods are normalised
  (`Q3 2026`, `2026Q3`, `3Q2026` -> `3Q26`; `FY2026` -> `FY26`; `1H26`) and must end up in one of those forms. Units use the
  guidance vocabulary (`%`, `USD`, `TWD billion`, ...); `currency` must match a money unit and be null otherwise; requires
  low <= mean <= high. Evidence `consensus:<symbol>`, kind `consensus`, payload tier `third-party aggregate`, event_time =
  available_at = observed `as_of`. A snapshot observed after the run's `as_of` is skipped with a note (not point-in-time).
- **Live probe** (`--consensus-live` or config `consensus_live: true`): stdlib-only Yahoo Finance. TW symbols map through
  config `adr_ticker` (`"UMC"` for the scope symbol, or `{"2303": "UMC", "2330": "TSM"}`); non-numeric symbols are used as is.
  Probe 1 = `https://finance.yahoo.com/quote/<T>/analysis/` and its server-rendered quoteSummary JSON (`earningsTrend`);
  probe 2 = `fc.yahoo.com` cookie + `getcrumb` + `quoteSummary?modules=earningsTrend`. Cloudflare challenge pages are refused.
  Every attempt `{url, status, bytes, sha256}` lands in `identity.consensus_status` (crumbs never recorded); any failure is
  `status: unavailable` with a reason and the run continues. Items: `eps_per_adr` (USD, per ADR — never matches a per-share
  actual) and `revenue` (TWD), period = calendar quarter/year of `endDate` (assumes calendar fiscal year), basis `n/a`.
  The raw page is saved to config `consensus_raw_dir` (default `<ledger dir>/consensus_raw/`).
  Observed 2026-10-21: with a browser User-Agent the UMC analysis page came back without the embedded quoteSummary (TSM had
  it); minutes later with the tool's own UA (`aquarium-research/1.0`) both UMC and TSM had it; `Python-urllib` UA got HTTP 429;
  `getcrumb` returned HTTP 429 every time. Cause of the UMC difference not established. Treat it as best-effort.
- **Surprise** (`surprise:<symbol>`, algorithm `surprise-v1`, derived, hashed): for each consensus item with an actual of the
  same metric (case/underscore-insensitive) and normalised period — actuals = document `actuals[]` + fact-claim `actual`.
  `surprise_abs = actual - mean` in the consensus unit (unit scale only; different currency/unit class -> `n/a (unit differs)`,
  never FX), `surprise_pct = surprise_abs / |mean| * 100` (2 dp), `vs_range` below/within/above `[low, high]`. A consensus
  observed after the actual was published (cited document `available_at`) gives `post-event`, no number. Only consensus for the
  config `symbol` is matched. Inputs are listed in the payload; `research_theme.replay_surprise(manifest_payload)` recomputes it.
- Claims citing `consensus:` / `surprise:` are numerically prechecked against the items / ok rows rendered as text.
- **Variant perception** (narrative `variant_perception[]`): `{metric, period, consensus_ref, our_view (number in the consensus
  unit | text), direction above|below|in_line, why}`. Refused unless `consensus_ref` is a frozen consensus evidence with that
  metric/period, `why` is non-empty, and a numeric `our_view` agrees with the direction (above/below vs mean; in_line = inside
  [low, high]). The period is stored normalised.
- **pilot_position** (config): `"long|short|none|undisclosed"` or `{position, size_note}` -> `research_json.disclosure`
  (hashed with the revision), printed in the disclosure block instead of 「未記錄」. Refused if the narrative also sets a
  different `disclosure`.
- **Supersede**: narrative `supersedes_revision_id` (or `supersedes`) must name a completed revision in the same ledger; rendered
  under 變更摘要 and consumed by `settle_theme.py` (SETTLEMENT.md).
- Render: "Consensus & surprise" (metric | period | consensus mean (range, n) | actual | surprise | vs range; unavailable
  probes listed below it) and "Variant perception", both after the guidance tables.
- Example: scratch re-run of the UMC note with the real Yahoo snapshot (observed 2026-10-21T04:13:20Z, raw sha
  `136fa6db…`) — no 3Q26 actual exists yet, so there is no surprise row; the Q3 revenue inferred from monthly revenue is a
  variant-perception `our_view`, not an actual.

## Vault knowledge-graph links (2026-10-21; design `local_dev/design/2026-10-21-vault-knowledge-graph-linking.md` §3.2–3.5)

- `project_vault.py` adds frontmatter edges `company`, `sector` (string for one target, list for several), `concepts`,
  `peers` (lists), `timeline`, plus `next_review_at` / `summary` / `catalysts` for `research_links.py`, and a trailing
  `## 知識連結` section with the same links. Concepts come only from the YAML blocks in `_kb/wiki/_taxonomy.md` (keyword
  match on thesis + claims + risks + invalidation_conditions; ASCII keywords case-insensitive on word boundaries).
  Sector scopes link every ohlcv symbol except `--benchmark` (default `0050`); `--config` adds its `read_across[]` to
  `peers`; `--company-slug SYMBOL=SLUG` names a company with no page yet. Missing targets get a stub page
  (`status: stub`, `origin: agent`); existing pages are never overwritten.
- `--replace` re-links an existing note only if `revision_id`, `content_sha256` and `origin: agent` match; every existing
  frontmatter line and the body stay byte-identical (the body is not re-rendered), only the edge fields and
  `## 知識連結` are regenerated.
- After each projection: `<vault>/_kb/ops/compile/research_links.py --apply` regenerates the company-page research
  timeline blocks and `timelines/<yyyy>-<symbol>` catalyst blocks, and `## [date] research | <symbol> <rev8> <stance>`
  is appended to `_kb/wiki/log.md`.

## Known gaps (tracked in SEED-002)

- Evidence acquisition is manual (download, pdftotext, choose line ranges). TSMC/MediaTek IR sites block
  fetches (Cloudflare); use SEC EDGAR or MOPS mirrors — the scout records `mirror_url` and `fetchable`.
- `price_facts:<symbol>` evidence (algorithm `price-facts-v2`) is derived from the frozen ohlcv payload and hashed, so the
  numbers in a report replay from the manifest; it is still computed by this script, not by Poseidon (B6).
- Acceptance lives in `agenda.jsonl` on macminim4, not in a Poseidon agenda table.
- Taiwan consensus has no free primary source; Yahoo coverage is via ADRs only (UMC/TSM), EPS per ADR, and Yahoo does not
  say which basis or which brokers. No median, no estimate history (a single snapshot per run).
