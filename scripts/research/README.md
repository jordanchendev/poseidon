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
- `themes` (2026-10-29 design §3.4): when claims carry `theme_ids`, `themes: ["[[themes/<id>]]", ...]` in claim order
  and one `- 主題：[[themes/<id>]]` bullet each under `## 知識連結`; an untagged revision has no `themes` key. Theme pages
  are never stubbed here: `_kb/wiki/themes/` belongs to the vault's `theme_pages.py`, and lint's `theme-id` rule checks
  every `themes/` link against `_kb/wiki/themes/_registry.md`.
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
  `response_format: json_object`, `reasoning.enabled` (default on; `--no-reasoning`), `max_tokens` 32000.
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
  [--bridge "ssh research-bridge"] [--agenda-json FILE] [--materials-json FILE] [--decisions-json FILE] \
  [--outbox DIR] [--agenda-id ID] [--dry-run] [--draft-model ID] [--no-reasoning] [--max-tokens N] [--jev-model ID] \
  [--narrative-apply off|on|confirm]
```

- Env files (600, checked; group/other-readable files are refused): `RESEARCH_DRAFT_OPENROUTER_KEY=...` and
  `JEV_RESEARCH_OPENROUTER_KEY=...` (`export` and quotes allowed). Keys are read only from these files, never from the
  process environment; a missing/empty key exits 3 with a message naming the variable, before the bridge is touched.
- Thalassa: `POSEIDON_THALASSA_BASE_URL` / `POSEIDON_THALASSA_API_KEY` from the process env (`docker --env-file`).
- `--dry-run`: assemble + freeze/prepare + prompt only, under `<state>/dry-run/` (own ledger), no keys, no model / JEV /
  bridge writes; writes `prompt.json`.
- Exit codes: 0 handled / nothing to do / lock busy, 3 key file, 4 a bridge report failed (retried next run),
  5 a bridge list call (`pending`, `material-pending`, `narrative-pending`) failed; the other stages still run.

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
theme_proposals, materials, narrative_records, narrative_events, narrative_checks, claim_theme_links), plus
`narrative_suggestions` for `--narrative-apply confirm` and `member_proposals`, `theme_proposal_claims` and
`held_records` for the owner's approvals (§9 step 9). Every row carries `content_sha256` of its other columns; a row
with a lifecycle (materials, theme_proposals, member_proposals, narrative_suggestions, held_records) is rehashed when
its status moves.

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
  `primary filing` or `company release`; `partially_verifies` states `supported_part` (the part of the record's claim
  the cited claims establish) and `open_part`.
- `[proposal]` at most 3 narrative proposals: atomic, numbers in the source claim's quote, a forecast's `resolves_on`
  year in a period the record names, an inference source with a stated basis; theme proposals well-formed.

`render_theme.py` adds 「敘事變化」 after 變更摘要 when the manifest's snapshot shows a record, from `narrative_results` in
revision.json (drafter verdicts only when there are none) and lists `disputed` / `missed` under 「待 owner 判斷」.

After the revision completes (§9 step 6), `narrative_check.apply` runs on every snapshot record: JEV judges the record
against the cited fact claims (quote and evidence texts) when the verdict is not `untouched`, and sweeps it against all
official excerpts in document order (12,000 characters, a cut source sets `sweep_truncated`). `RULE_ROWS` is the §4.5
table R1 to R9 keyed by (claim_type, verdict, label, basis_ok, before_resolution, settled); a key no row covers is
`no_change`, and a status the rule may not reach under `TRANSITIONS` is `disputed`. Before the table, a drafter and a
sweep that disagree in direction (verifies / partially_verifies against a sweep that contradicts, refutes against a
sweep that supports; `SWEEP_DISAGREES`) are R6 `disputed`. A state `partially_verifies` gets two more judgements of
the drafter's `supported_part` (owner decision 2026-10-30, after ns-712a8221 and ns-06ab94e2): against the record's
claim (`in_record`, period precheck only: it must be part of the record) and against the same cited source (`backed`).
R3 takes it only when both support; a `backed` (or whole-claim) contradicts is R6, anything else is `no_change` with a
`why` naming the failed judgement, shown in 「敘事變化」. The event or suggestion keeps `supported_part` and
`open_part` in basis and both judgements under jev `supported_part`; the confirm line shows both parts. One
`narrative_checks` row per
record always; a `status` or `corroborated` event only with `runner.py --narrative-apply on` (default `off` until the
§9 step 7 control set passes), at max(evidence available_at, the record's last event at). A revision whose check rows
exist is skipped without JEV; an event that already names the revision is reused. The results go to
`revision.json.narrative_results`; a failure is `run.json.narrative_error` and the revision stays completed.
Before the records it writes one `claim_theme_links` row per (claim index, theme_id) the drafter tagged
(`source` drafter, `about_symbols`, the claim kind, and the tier and event_time of the claim's highest-tier cited
evidence), whatever `--narrative-apply` says: a link is not a status change (D2). A key already written is kept, so a
rerun writes nothing new and a rerun after a crash fills in what is missing. A revision's `theme_proposals` are filed
in every mode too (below, "Theme and member approvals"). So are its `narrative_proposals` (design §3.2,
`write_narrative_proposals`): each becomes a record through intake's path, origin `revision:<uuid>#claims[i]`, the
cited claim's quote verbatim in the evidence text it comes from, `first_seen_at` = the revision's as_of. A fact claim
makes an `official` record (attributed to the document's publisher) only when every evidence it cites is official
(§6.1: secondary material never makes fact evidence); an inference claim makes `our-inference` (本系統推論). A subject
that only an open theme or member proposal names is held (`held_records`) until the owner decides it; anything else
the ledger refuses is listed as unfit. `narrative_results.narrative_proposals` [{index, narrative_id, claim, status
written|held|unfit, waits, reason}] is rendered as 「新增敘事」 in the done report.

- Recall keywords (§5.3): a theme proposal (revision or intake) may suggest up to 8 `keywords` (each at most 40
  characters, kept in `theme_proposals.keywords_json`); accepting the theme copies them. The owner adds more with
  `narrative_ledger.py theme-keywords --ledger L --id <theme> --add <kw> [...] --approval-ref <ref> [--at T]`:
  `approval_ref` is required, each new keyword gets a `theme_keyword_approvals` row (theme, keyword, approval), the
  theme's `keywords_json` is appended and its row hash recomputed, and a keyword the theme has is a no-op. A ledger
  made before this gets the `keywords_json` column on first connect, its proposal rows rehashed. After each
  completed revision with a snapshot, a claim whose text hits a keyword of a shown theme it is not tagged with
  (`project_vault.keyword_hit`, the vault's concept rule) is listed in `narrative_results.untagged_hits`
  [{claim_index, theme_id, keyword}] and rendered as 「主題召回」 in the done report. A keyword never makes a link.
  The owner tags such a claim with `narrative_ledger.py link-add --ledger L --revision <id> --claim <index> --theme
  <theme> [--about <symbol> ...] --approval-ref <ref> [--at T]`: the revision must be completed in the same ledger
  with its manifest; the `claim_theme_links` row has `source` owner, the claim's kind, and the tier and event_time of
  its best cited evidence (as the drafter's links); `--about` defaults to the claim's `about_symbols` and is required
  when it has none; a `claim_link_approvals` row keeps the approval. The same link again is a no-op.
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
  number missing from the quote (`check_numbers`), and needs each subject to be an approved member of one of the
  record's themes.
- `export --as-of T` is a point-in-time view (events with `at <= T`), sorted, quotes clipped to 120 characters (D7),
  written tmp + rename with `--out`. `narrative-export-v2` (§9 step 10) adds, per `claim_theme_links` row, the
  revision's `scope_key`, `claim_text` and `claim_quote` (clipped; from the completed `research_revisions` row in the
  same ledger, null when absent), `signal_window_days` (180, D11) and `cross_company_signals` [{theme_id, level
  official|relayed, since, companies [{symbol, first_at, last_at, n}]}], computed in SQL (`SIGNAL_SQL`, design §5.2):
  official = fact claims linked to the theme on `primary filing` / `company release` evidence whose `about_symbols`
  cover two or more companies with evidence event_time in (as_of - 180 d, as_of], from documents of two or more
  issuing companies (the payload issuer of the evidence each link describes, read from the revision's manifest in the
  same ledger; assemble records the row's one subject symbol, else its spot symbol, as the issuer; the URL host is
  not the issuer; a link without one is no hit), so one company's documents never make the signal alone; relayed =
  sell-side / media / kol records of the theme not refuted at as_of whose `subject_symbols` cover two or more
  companies with `first_seen_at` in the window, listed only when the theme is not an official signal. A symbol the
  owner rejected from a theme (a `rejected` member proposal decided by as_of, and no approved member row) is no hit
  for that theme in either tier and is left out of that theme in the export (dropped from the about_symbols of its
  links, a link left with none omitted; a record whose every subject was rejected loses that theme), however old the
  link or record; the `claim_theme_links` and `narrative_records` rows are kept. Other themes are unaffected.
- The runner rewrites `<state-dir>/narrative_export.json` (export as of now) after every invocation but a dry run;
  `nexus_pull.py` on the MacBook pulls it (no host wrapper change). A failure is logged and never changes the exit code.
- `themes.seed.json` is the §7.1 seed, approved by the owner on 2026-10-29 (`six_approval_ref`), with each member's official source. Members under
  `candidate` are not approved (D5); nothing loads the seed into a ledger except the test's tmp ledger. Since
  themes-seed-v2 it also carries the four themes the owner accepted in the trial ledger on 2026-10-30 (high-na-euv,
  custom-asic-capex, xpu-interconnect-content, hbm-base-die-foundry); each of them and each of their members has
  `approval` (the Telegram decision ref and time), and their members list the fields no proposal recorded under
  `unrecorded`. NASDAQ:AMD (customer, M20261029-2) is the one such member inside a §7.1 theme (scale-up-optics). The intake and theme-decision tests load only the six §7.1 themes (test_intake.SECTION_7_1).

Tests: `PYTHONPATH=src python -m pytest -q -p no:cacheprovider scripts/research/test_narrative_ledger.py`.

### Material intake (§9 step 8, `intake.py`)

The owner pastes a KOL post, media article or sell-side note in Telegram; Sisyphus runs
`research-agenda.sh submit-material --tier sell-side|media|kol --author A [--url U]` with the text on STDIN, and
macminim4 keeps it (`~/.openclaw/research/materials.jsonl` + `materials/<id>.txt`, id `YYYYMMDD-kN`, the same text
ignoring whitespace within 24 h is not taken twice). Each runner invocation takes in at most one material from the
bridge's `material-pending` (`--materials-json` in host mode):

- The row's text must hash to `text_sha256` (the bridge already refuses to hand over a text that does not; it sends
  `text_error` instead, and the runner reports needs-review).
- Freeze: `runs/material-<id>/sources/material.txt`, one excerpt per non-empty line, evidence
  `secondary:material-<id>` (kind `secondary`, the material's tier), plus a `narrative_snapshot` with every active theme
  (records `[]`) when there is one; `ManifestService.freeze` writes `data_manifests` only (market `tw_stock`, interval
  `material`, required `secondary` within 30 days). Never a `research_revisions` row.
- Draft (same key and model as the research draft, D9): `records[]` {claim, claim_type, resolves_on, attributed_to,
  quote, subject_symbols, theme_ids}, `theme_proposals[]` {suggested_id, title, kind, why, quote, proposed_members},
  `member_proposals[]` {theme_id, symbol, role, why, quote} (§9 step 9), `verify_with[]`.
- Gates, one repair round: quote verbatim in the material; `lint_atomic`, and no causal or driver connector
  (`claim_precheck._CAUSAL`: 帶動, 由於, driven by …), because a record is checked later one statement at a time;
  `check_numbers` of the record against its quote; a forecast's `resolves_on` names a year of the record (else null:
  the ledger default); theme ids from the closed list, or the `suggested_id` of a theme proposal filed in the same
  output; each subject an approved member of one of the record's themes, or a member a theme or member proposal filed
  in the same output names into one of them (a member of another theme does not count; a proposal that repeats an
  active theme files no member of its own; one that repeats an open proposal admits that proposal's members; one the
  owner rejected admits nothing); member proposals name a listed theme; at most 10
  records, 3 theme proposals, 3 member proposals, 5 `verify_with`. Refused twice -> `material-needs-review`.
- Faithful-paraphrase check: `narrative_check.judge_record` (period precheck, then the 3-vote JEV majority) of each
  record against the material's heading line (its first non-empty line) and the line(s) its quote sits in: a post's
  lines often do not name the company its heading names. Anything but `supports` drops the record and lists it.
- Writes: `theme_proposals` (`T<YYYYMMDD>-n`, source_ref `material:<id>#p<line>`; a title or keyword that repeats an
  active theme, or the id or title of an open, accepted or rejected proposal, is listed and not filed) and
  `member_proposals` (`M<YYYYMMDD>-n`; an approved member or an open or rejected proposal of the same member is listed
  and not filed); then `narrative_records` with `source_tier` = the material's tier, `first_seen_at` = `received_at`,
  origin `material:<id>`, and the seq-0 `created` event. A record that needs a proposal still pending is held
  (`held_records`, below) instead; `materials` row `intaken` with the manifest sha. `intake.json` keeps the checked plan, so a rerun
  after a crash writes the same rows without a model call; a material the bridge still lists after it was reported is
  reported again from `run.json`.
- `material-done <id>` stdin JSON `{records, held, dropped, theme_proposals, member_proposals, verify_with, text}`;
  `text` is the Telegram notice (new records, held records with what they wait for, dropped records with the JEV label
  or the rejected proposal, theme and member proposals with their ids and how to answer, duplicates, documents to
  verify with; a material with nothing to track says 「無對應主題」).

### Owner confirm mode (`--narrative-apply confirm`, owner decision 2026-10-29)

For the first 4 weeks the rule table's changes are suggestions: JEV still calls two contradicts cases of the control
set supports at >= 0.95. `off` and `on` behave as before.

- Each event the rule would write (status change or corroboration) becomes a `narrative_suggestions` row
  (`ns-` + 8 hex of the revision and record ids). The check row is written as in `off`. The result carries
  `suggestion` {id, status, line}; `done` stdin JSON adds `suggestions` [{id, narrative_id, line}], and the bridge
  appends each line (「建議 ns-… … 由 未證實 改為 部分證實（R3），依據 claims[i]「…」；撐住的部分「…」；未撐住的部分「…」；回覆
  「確認敘事 ns-…」或「駁回敘事 ns-…」」, the two parts only for a partially_verifies)
  to the notice. render_theme.py lists them under 「待 owner 確認」.
- The owner replies; Sisyphus runs `research-agenda.sh narrative-confirm|narrative-reject <ns-id>` (or
  `narrative-set <nr-id> <status>`, the §4.6 override) with the owner's sentence on STDIN; macminim4 checks the id
  against the suggestions it was sent and appends a pending decision (`YYYYMMDD-dN`) to `narrative_decisions.jsonl`.
- Next invocation: `narrative-pending` (`--decisions-json` in host mode) -> `narrative_check.apply_decision`:
  confirm writes the held event with `decided_by` owner, the suggesting rule, `note` = the owner's sentence and
  `basis.approval_ref` = `telegram:<decision id>`, at max(decision time, the record's last event); reject marks the
  suggestion, and the same change to that record is not suggested again (the result shows `suppressed_by`); set is
  `set_status` with the same approval_ref. Each decision is acked with `narrative-ack <id> applied|refused` and a
  zh-TW message the bridge sends to Telegram. A decision delivered again is a no-op (`（先前已套用，未重複寫入）`).
- One piece of evidence corroborates a record once (owner decision 2026-10-30, after ns-a9165163 / ns-dd5e60a8): with
  mode on or confirm, a corroborated event or suggestion is not written when every cited evidence already backs a
  corroborated event or a pending or confirmed corroborated suggestion of the record from another revision. An
  evidence is the sha256 of its raw files (a document; its id and content_sha256 change on a re-freeze), else of its
  payload; basis keeps the list as `evidence_sha256`, and an older row without it counts when every quote it stored is
  in the cited evidence's texts. The check row is written; the result shows `duplicate_of` (mode on) or a suppressed
  `suggestion` with `duplicate_of` (confirm), and render_theme.py says 「同一份證據已背書過 …，不再寫入」. A rejected
  suggestion still suppresses first (`suppressed_by`).

### Host wrapper (proposed; the live `~/research-runner/run.sh` is the owner's)

```bash
ssh research-bridge pending > $S/pending.json.tmp && mv $S/pending.json.tmp $S/pending.json
ssh research-bridge material-pending > $S/materials.json.tmp && mv $S/materials.json.tmp $S/materials.json
ssh research-bridge narrative-pending > $S/decisions.json.tmp && mv $S/decisions.json.tmp $S/decisions.json
# skip the container only when all three lists are empty
docker run ... scripts/research/runner.py --state-dir /state --agenda-json /state/pending.json \
  --materials-json /state/materials.json --decisions-json /state/decisions.json --outbox /state/outbox ...
python3 scripts/research/deliver_outbox.py --outbox $S/outbox --bridge "ssh research-bridge"
```

In host mode (`--agenda-json`) a list the host did not pass is empty: the container never calls ssh. `deliver_outbox.py`
delivers `done`, `needs-review`, `material-done`, `material-needs-review` and `narrative-ack` entries.

### Theme and member approvals (§9 step 9; design §3.3)

Agents propose; only the owner's decision writes `themes` or `theme_members`, and the id of a new theme is the string
the owner typed.

- Proposals. A completed revision files each of its `theme_proposals` (`narrative_check.file_theme_proposals`, every
  `--narrative-apply` mode, source_ref `revision:<uuid>#claims[i,...]`) and stores the claims it cites in
  `theme_proposal_claims` (about_symbols = the claim's, else the revision's subjects; evidence_tier and event_time of
  the claim's highest-tier evidence). A proposal that repeats an open one adds its claims to it; one that repeats an
  approved theme or a rejected proposal is only recorded in `narrative_results.theme_proposals`. The `done` stdin JSON
  adds `theme_proposals` [{id, line}] and the bridge appends each line to the notice:
  「主題提案 T…：<title>（<kind>），依據 claims[i]；建議 id <suggested>；提案成員 …；回覆「接受主題 T… <id>」或「略過主題 T…」」.
  Intake files theme and member proposals as above and lists them in the material-done notice.
- Held records. An intake record tagged with a theme proposal, or whose subject is not yet an approved member of one
  of its themes, is kept in `held_records` (the `add_record` arguments, the quote's line(s), and the open proposal ids
  it waits for), never in `narrative_records`. A subject is covered by an open member proposal of the symbol into one
  of the record's themes, or by an open theme proposal the record is tagged with that names it; a proposal into
  another theme does not count, so a member the owner rejected from the record's theme drops the record (「NASDAQ:ASML
  不是 cleanroom-capacity 的核准成員（成員提案 M… 已被略過）」) even when another theme proposes the symbol. Every need must be met and any one proposal meets its need, so notices write the waits as
  「M… 或 T…」 within a need and 「，以及」 between needs (`nl.waits_text`).
- Decisions. Sisyphus runs `research-agenda.sh theme-accept <T-id> <theme-id>`, `theme-reject <T-id>`,
  `member-accept <M-id>` or `member-reject <M-id>` with the owner's sentence on STDIN; macminim4 appends a pending
  decision to the same `narrative_decisions.jsonl`, and the runner reads it from the step 8 feed `narrative-pending`
  (`--decisions-json` in host mode). One feed instead of the design's `theme-pending`: the host wrapper, the outbox and
  the ack verb already exist for it, and decisions keep one order.
  - theme-accept: the id must match `^[a-z0-9]+(?:-[a-z0-9]+)*$` (else refused and acked refused); `themes` row with
    title, kind and suggested `keywords` of the proposal, `approval_ref` `telegram:<T-id>`, `proposal_id`;
    `claim_theme_links` with
    `source` owner for every claim in `theme_proposal_claims`; a member proposal for each proposed member (accepting a
    theme approves no member). Accepting again under the same id is a no-op; under another id, or after a reject, is
    refused.
  - member-accept: `member-add` with `approval_ref` `telegram:<M-id>`.
  - theme-reject / member-reject: the proposal is `rejected` and never filed again (`theme_duplicate` /
    `member_duplicate` report `rejected:<id>`).
  - After each of these the runner re-gates every held record (`nl.release_held`): written with its original tier and
    `first_seen_at` when nothing it needs is pending, dropped with the reason when something it needs was rejected,
    otherwise kept with refreshed waits. The `narrative-ack` message says which records moved.

Tests: `test_intake.py` (the §7 posts in `fixtures/gooaye_posts.json`, a test fixture only, never vault content),
`test_narrative_confirm.py` and `test_theme_decisions.py`; the macminim4 side is
`local_dev/scripts/strategy-bridge/macminim4/test_research_intake.py` in the aquarium workspace.

### Vault projection of themes (§9 step 10; design §3.4, §5.1-§5.2)

The ledger stays the only source; the vault gets a one-way projection on the MacBook. The vault-side scripts are not
in this repo: proposed copies live in the aquarium workspace under `local_dev/scripts/research/nexus/`, laid out like
the vault (`_kb/ops/compile/theme_pages.py` new; `research_links.py`, `_kb/ops/lint/lint.py` and `bin/nexus_pull.py`
changed, each changed line marked `# [themes]`), for the owner to install.

- `nexus_pull.py` pulls `~/research-runner/state/narrative_export.json` when its sha256 changed (sensitive patterns
  quarantine it like any other pull), runs `theme_pages.py --export <cache> --apply`, then `research_links.py --apply`,
  `compile.py --apply`, `lint.py --write`.
- `theme_pages.py` writes `_kb/wiki/themes/_registry.md` whole (table, theme edges, and a YAML block `themes:` with
  title, kind, status, approval, keywords, members and `linked_symbols`) and, per theme, creates
  `_kb/wiki/themes/<id>.md` once as a stub; afterwards only the block between `<!-- theme-timeline:start ... -->` and
  `<!-- theme-timeline:end -->` is regenerated: 跨公司訊號 (the export's signals, official and relayed apart), 時間線
  (官方事實 / 官方預期 / 本系統推論 / 傳聞（層級）/ 狀態變化 when the status moved / 官方背書 for a corroborated
  event, oldest first, Research notes linked by scope) and 敘事追蹤.
  Quotes are at most 120 characters and no material text is in the export (D7). The same export gives no write.
- `lint.py` rule `theme-id` (ERROR): a `[[themes/<id>]]` link, or a `themes` frontmatter value, whose id is not in the
  registry's YAML block, or any `themes/` link when the registry is missing. `research_links.py` starts a company block
  with one `相關主題：` line when the page's symbol (+ market for non-Taiwan codes) is a member or a linked symbol.

Tests: `test_theme_pages.py` runs the vault scripts from `$NEXUS_VAULT_SCRIPTS` against a fixture vault through
`NEXUS_VAULT` (they skip when the scripts are absent; its export tests always run); with `NEXUS_VAULT` and
`NEXUS_VAULT_SCRIPTS` pointing at the nexus copies, `test_vault_lint_graph.py` and the research_links tests in
`test_offline_tooling.py` run against them too.
