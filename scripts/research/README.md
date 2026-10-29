# Research loop stopgap tooling (2026-10-20)

> Moved 2026-10-21 from aquarium `local_dev/scripts/research/` into Poseidon `scripts/research/` (versioned). Paths below
> that start with `local_dev/scripts/research/` now mean `scripts/research/` in this repo; tests run from the repo root:
> `PYTHONPATH=src python -m pytest -q -p no:cacheprovider scripts/research/`. The server-side runner (`runner.py`) is
> described in "Runner on stormtrooper" at the end of this file.

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
- **Read-across** — config `read_across[]` is refused (design 2026-10-29 §2.7): a claim names the companies it is
  about in its `about_symbols`. Revisions made before 2026-10-29 keep `identity.read_across: {symbol, relation, why}`
  and still render it (T+1/T+5 relative returns from price_facts when the symbol is in `symbols`, else n/a).
- **Precheck split** — fact claims: any numeric mismatch refuses. Inference claims: only scale/sign refuse; other
  mismatches are recorded as `numbers: mismatch-recorded`.
- **Disclosure** — last rendered section: origin, provider/model/policy, each evidence with its tier
  (`documents[].source_tier` overrides, one of primary filing | company release | exchange feed | derived | sell-side |
  media | kol; otherwise sec.gov/MOPS -> primary filing, other documents -> company release,
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
  Sector scopes link every ohlcv symbol except `--benchmark` (default `0050`); `peers` come from an old revision's
  `identity.read_across`; `--company-slug SYMBOL=SLUG` names a company with no page yet. Missing targets get a stub page
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

## Runner on stormtrooper (2026-10-21)

Design: aquarium `local_dev/design/2026-10-21-research-runner-stormtrooper.md`. One accepted agenda topic per invocation,
no human in the loop; nothing is written to Poseidon production (isolated SQLite ledger until the 101-23 soak gate).

```
pending row ─ assemble.py ─ research_theme.py (freeze + prepare) ─ draft.py (OpenRouter) ─ gates ─┬─ JEV ─ complete ─ done
                                                                                                 └─ refused: ONE repair round, then needs-review
```

- **assemble.py** (no LLM): fetches `primary_source_url` (falls back to `mirror_url` on a fetch error or when no excerpt
  matched) with `fetch_source.build_document`; template from `event_type` (earnings / investor_conference -> earnings,
  press_release -> press_release, else generic); keeps every verified proposal in document order up to 40. Daily bars for
  the Taiwan symbols + `0050` from Thalassa (Poseidon `RemoteDataRepository`, env `POSEIDON_THALASSA_BASE_URL` /
  `POSEIDON_THALASSA_API_KEY`), `kpi_series:<symbol>` from TWSE t187ap05_L (a TWSE failure only drops kpi_series).
  Event/document time = `event_date`T00:00Z, clamped to as_of when the event is in the future. Freshness rules:
  fundamental 90 days, ohlcv 5 days (so Monday-morning runs see Friday's bar). Non-numeric symbols get no OHLCV.
  One TW symbol -> scope `tw_stock:spot:<sym>`; several -> `tw_stock:basket:<a>-<b>`.
  When no fetched candidate matches the template (e.g. a MOPS notice of a future 法說會 for an earnings topic), the
  first fetched document falls back to the generic template (paragraphs >= 20 chars, the longest 40 kept in document
  order, pre-verified verbatim) instead of failing; an unfetchable source (primary and mirror) still fails.
- **event_status** (owner decision 2026-10-21; assemble.py `event_status`, never the model): `pre_event` when
  `event_date` is after as_of, OR the document dates the event after as_of (value after 事實發生日 / 召開…日期 /
  舉行…時間 / 會議…日期, ROC years +1911; the latest such date becomes `event_date`), OR no template excerpt matched
  (the outcome is not in the evidence); else `post_event`. Stored as config `event` {status, event_date, why[]}, shown
  to the model in the research context, and written into the narrative (`event_status`, `event_date`) after every
  draft, so it lands in research_json. pre_event: the prompt says the note is a pre-event expectation and every
  outcome claim is an inference with its basis; gates refuse an empty/missing `catalysts` and (when event_date is on or
  after as_of) a catalyst list without the event date. render_theme.py opens a pre_event revision with
  `> [!warning] 事前推論：事件（<date>）尚未發生…` (event date before as_of: `證據中沒有事件（<date>）的結果…`);
  project_vault.py writes `event_status` to the frontmatter (machine field, like next_review_at).
- **draft.py**: OpenRouter chat completions, model from config (default `deepseek/deepseek-v4.1-flash-1017`),
  `response_format: json_object`, `reasoning.enabled` (default on; `--no-reasoning`), `max_tokens` 16000.
  `content=None` or `finish_reason=length` is a draft failure. Records configured model, returned model, prompt sha256,
  finish_reason, usage and `usage.cost`. Narrative schema = the research_theme schema (`tl_dr`, `thesis`, `stance`
  watch|neutral|hold|long|short, `change_summary`, `next_review_at`, optional `catalysts`, `risks`,
  `invalidation_conditions`, `unknowns`, `claims[]`) where each claim is `{kind fact|inference, text, evidence_ids[],
  quote}` and `quote` must be copied verbatim from one cited evidence text (document excerpt, or a line of the rendered
  kpi / price_facts / ohlcv block shown in the prompt). `quote` is stored in research_json (Poseidon keeps extra claim keys).
- **Gates** (runner.py `run_gates`): schema (incl. Poseidon `validate_research`: >= 3 claims, required fundamental + ohlcv
  cited), `validate_narrative_extras` / `validate_variant_perception`, quote-in-cited-evidence, `precheck_claims` with
  `allow_non_atomic=False` (fact: any numeric mismatch refuses; inference: scale/sign, also against an A-B/A+B
  derivation; a recomputed derivation is recorded as `derived` [{claim, how, from, value}]), lint_atomic (inference
  claims: the multi-quantity rule does not refuse, causal clauses still do), GAAP basis, inference basis marker
  (`claim_precheck.has_inference_basis`: 推算 估算 推估 推論 估計 判斷 研判 預期 預估 可能 若 如果 假設 換算 計算 相減
  相加 合計 依(not 依然/依舊) 根據 基於 意味 暗示 隱含 有望 或將 恐將 / based on, implies, suggests, likely, probably,
  may, might, could, expect, estimate, if, assuming, derived, computed, calculated), pre_event catalysts. Inference
  claims render with a `【推論】` prefix (facts unprefixed) in render_theme.py. Refusal ->
  one repair round (the model gets its previous JSON and the refusal list) -> gates again. JEV `typesafe/jev-1.13`
  (`/v1/systemone`, falls back to `/alpha/decisions` on 404) judges **fact** claims only (inference claims are
  never sent) against the claim quote + the cited evidence texts (capped at 12k chars); every label must be `supports`. Completion runs
  research_theme.py with `--narrative`, which re-runs its own prechecks, completes, and checks replay + immutability.
- **Outcome**: completed -> bridge `done <agenda_id> <revision_id>` with the tl_dr on stdin. Anything else (assemble,
  freeze, draft, gates after repair, JEV, completion, internal error) -> revision stays `prepared`, bridge
  `needs-review <agenda_id>` with `<stage>: <reasons>` (<= 580 chars) on stdin. No escalation to another model.
- **Idempotency**: `runner_runs` in the ledger. A row already completed / needs_review that the bridge still lists as
  accepted is reported again (no new run, no model call). A row left `running` by a crash is rerun. To force a rerun,
  delete its `runner_runs` row.

### Invocation

```bash
python scripts/research/runner.py --state-dir ~/research-runner/state \
  --draft-env ~/research-runner/secrets/research-draft-openrouter.env \
  --jev-env ~/research-runner/secrets/jev-openrouter.env \
  [--bridge "ssh research-bridge"] [--agenda-json FILE] [--outbox DIR] [--agenda-id ID] [--dry-run] \
  [--draft-model ID] [--no-reasoning] [--max-tokens N] [--jev-model ID]
```

- Env files (600, checked; group/other-readable files are refused): `RESEARCH_DRAFT_OPENROUTER_KEY=...` and
  `JEV_RESEARCH_OPENROUTER_KEY=...` (`export` and quotes allowed). Keys are read only from these files, never from the
  process environment; a missing/empty key exits 3 with a message naming the variable, before the bridge is touched.
- Thalassa: `POSEIDON_THALASSA_BASE_URL` / `POSEIDON_THALASSA_API_KEY` from the process env (`docker --env-file`).
- `--dry-run`: assemble + freeze/prepare + prompt only, under `<state>/dry-run/` (own ledger), no keys, no model / JEV /
  bridge writes; writes `prompt.json`.
- Exit codes: 0 handled / nothing to do / lock busy, 3 key file, 4 a bridge report failed (retried next run),
  5 bridge `pending` failed.

Outputs under `--state-dir`: `ledger.sqlite` (research_theme tables + `runner_runs`), `runner.log`, `runner.lock`,
`runs/<agenda_id>/` with `config.json`, `bars.csv`, `sources/` (raw + text + `twse-t187ap05_L.json`), `narrative.json`
(+ `narrative.draft1.json` after a repair), `revision.json` (`identity`, `research`, `manifest_payload`, `jev_results`; on
needs_review `research` is null, `identity.status` is `prepared` and `needs_review` lists the reasons) and `run.json`
(status, failed_stage, reasons, timings, models configured/returned, draft calls with prompt sha256 / usage / cost, JEV
usage, costs, gate reports).

### docker run (sketch, not run here)

The `poseidon-cpu-worker` image (python:3.13-slim) has everything the runner imports (pandas, sqlalchemy, httpx,
tenacity, poseidon) but, most likely, no `ssh` client and no `pdftotext`:

- no ssh -> the host does the bridge I/O: pull `pending` into a file, run the container with `--agenda-json` and
  `--outbox`, then deliver the queued reports with `deliver_outbox.py` (stdlib, Python 3.8+) on the host.
- no pdftotext -> a PDF source fails assemble with "needs pdftotext (poppler-utils)" -> needs-review. HTML sources work.

```bash
set -e; cd ~/Projects/poseidon-research; S=~/research-runner
ssh research-bridge pending > $S/state/pending.json.tmp && mv $S/state/pending.json.tmp $S/state/pending.json
docker run --rm --user "$(id -u):$(id -g)" --env-file $S/secrets/thalassa.env \
  -v $PWD/src/poseidon:/app/.venv/lib/python3.13/site-packages/poseidon:ro -v $PWD/scripts:/app/scripts:ro \
  -v $S/state:/state -v $S/secrets:/secrets:ro -w /app poseidon-cpu-worker \
  python scripts/research/runner.py --state-dir /state --draft-env /secrets/research-draft-openrouter.env \
    --jev-env /secrets/jev-openrouter.env --agenda-json /state/pending.json --outbox /state/outbox
python3 scripts/research/deliver_outbox.py --outbox $S/state/outbox --bridge "ssh research-bridge"
```

`thalassa.env` holds only the two `POSEIDON_THALASSA_*` values (not the compose `.env`: Settings() cannot parse its JSON
values). Mount `scripts/` read-only: the runner writes only under `--state-dir`.
Tests: `PYTHONPATH=src python -m pytest -q -p no:cacheprovider scripts/research/test_runner.py` (offline; the bridge,
OpenRouter, JEV, Thalassa and TWSE are faked; the source is served from a file:// URL through fetch_source).

## Theme and narrative ledger (2026-10-29; design `local_dev/design/2026-10-29-research-theme-and-narrative-ledger.md`)

`narrative_ledger.py` keeps nine tables in the runner's ledger file (themes, theme_members, theme_edges,
theme_proposals, materials, narrative_records, narrative_events, narrative_checks, claim_theme_links). Every row carries
`content_sha256` of its other columns.

What the runner reads from it (design §9 step 4, `assemble.py`; the ledger is opened read-only, a dry run included):

- Agenda symbols: a bare code is Taiwan-listed and priced; `<EXCHANGE>:<TICKER>` goes to `cfg.subject_symbols` with no
  OHLCV. A bare code that the ledger knows only as a non-Taiwan member's ticker (`3110` vs `TSE:3110`) is refused.
- A row with `theme_id` researches `tw_stock:theme:<id>`: OHLCV is the theme's Taiwan-listed members approved by as_of
  plus 0050. A Taiwan code in the row must be one of them; a theme with no Taiwan-listed member is refused (D4).
- `cfg.narrative_snapshot` (§4.1): the run's themes (a theme scope's own theme, and every active theme with one of the
  run's symbols as a member approved by as_of) and open records (`unverified`, `partially_verified`, `stale`, folded
  at as_of, `first_seen_at <= as_of`) whose subjects meet the run's symbols or whose themes meet the run's themes;
  subject hits first, newest first, at most 15, the rest in `not_shown`. A member run with no open record still gets
  one, with `records: []`, because the theme list for tagging travels only in it. `research_theme.py` freezes it last
  as evidence `narrative_snapshot`. A run that meets no theme and no open record gets no snapshot: the manifest and
  prompt are byte-identical to before (pinned in `test_narrative_snapshot.py`).
- `documents[]` of kind `secondary` need `source_tier` sell-side, media or kol. `claim_precheck.evidence_tier` is the
  one tier mapping, for the rendered disclosure and the narrative gates.

Drafting against a snapshot (§9 step 5): `draft.py` adds the closed theme list and the open records to the evidence
block, `NARRATIVE_LEDGER_SCHEMA` (claim `theme_ids` / `about_symbols`, `narrative_checks`, `narrative_proposals`,
`theme_proposals`) and rules 11 to 14, only when the manifest has a snapshot. `narrative_check.gates` runs inside
`run_gates` (one repair round, as the other gates):

- `[evidence]` a fact claim never cites `narrative_snapshot` or `secondary` evidence.
- `[theme]` `theme_ids` from the snapshot's list, with `about_symbols` from the run's symbols, subjects or theme members.
- `[check]` exactly one check per shown record; a verdict other than `untouched` cites fact claims whose evidence is all
  `primary filing` or `company release`; `partially_verifies` states `open_part`.
- `[proposal]` at most 3 narrative proposals: atomic, numbers in the source claim's quote, a forecast's `resolves_on`
  year in a period the record names, an inference source with a stated basis; theme proposals well-formed.

`render_theme.py` adds 「敘事變化」 after 變更摘要 when the manifest's snapshot shows a record, from `narrative_results` in
revision.json (drafter verdicts only when there are none) and lists `disputed` / `missed` under 「待 owner 判斷」.

After the revision completes (§9 step 6), `narrative_check.apply` runs on every snapshot record: JEV judges the record
against the cited fact claims (quote and evidence texts) when the verdict is not `untouched`, and sweeps it against all
official excerpts in document order (12,000 characters, a cut source sets `sweep_truncated`). `RULE_ROWS` is the §4.5
table R1 to R9 keyed by (claim_type, verdict, label, basis_ok, before_resolution, settled); a key no row covers is
`no_change`, and a status the rule may not reach under `TRANSITIONS` is `disputed`. One `narrative_checks` row per
record always; a `status` or `corroborated` event only with `runner.py --narrative-apply on` (default `off` until the
§9 step 7 control set passes), at max(evidence available_at, the record's last event at). A revision whose check rows
exist is skipped without JEV; an event that already names the revision is reused. The results go to
`revision.json.narrative_results`; a failure is `run.json.narrative_error` and the revision stays completed.
Before the records it writes one `claim_theme_links` row per (claim index, theme_id) the drafter tagged
(`source` drafter, `about_symbols`, the claim kind, and the tier and event_time of the claim's highest-tier cited
evidence), whatever `--narrative-apply` says: a link is not a status change (D2). A key already written is kept, so a
rerun writes nothing new and a rerun after a crash fills in what is missing. `narrative_proposals` and
`theme_proposals` stay in revision.json; §9 steps 8 and 9 write them to the ledger.

- Themes, members and edges are written only with a non-empty `--approval-ref`; re-running the same approval is a
  no-op, the same key with different content is refused. Symbols: Taiwan-listed bare (`2408`), others
  `<EXCHANGE>:<TICKER>` (`NASDAQ:MU`, `TSE:3110`); `tw_listed` follows from that shape.
- A narrative's status is the fold of its append-only events (`created`, `status`, `override`, `corroborated`, the last
  one never changes the status). `TRANSITIONS` is the design §2.4 table: rule, sweep and owner each have their own
  allowed moves; `set-status` is the owner override and needs `--note`.
- `review_by` (D6): state claims `first_seen_at` + 180 days; forecasts `resolves_on` + 90 days, and a forecast without
  a period gets `resolves_on` = `first_seen_at` + 365 days (`resolves_on_default = 1`). `sweep-stale --as-of T` moves
  only `unverified` records past `review_by` to `stale`.
- `record-add` refuses a quote that is not verbatim in `--source-file`, a non-atomic claim (`lint_atomic`) and a claim
  number missing from the quote (`check_numbers`), and needs subjects that are approved members.
- `export --as-of T` is a point-in-time view (events with `at <= T`), sorted, quotes clipped to 120 characters (D7).
- `themes.seed.json` is the §7.1 seed for the owner to review, with each member's official source. Members under
  `candidate` are not approved (D5); nothing loads the seed into a ledger except the test's tmp ledger.

Tests: `PYTHONPATH=src python -m pytest -q -p no:cacheprovider scripts/research/test_narrative_ledger.py`.
