# 2330 research PoC

Two historical TSMC snapshots combine official financial disclosures and stored daily OHLCV. The host supplies read-only MCP tools, validates citations and disclosure cutoffs, and saves immutable research revisions in SQLite. Replaying identical evidence/policy/provider/model reuses the revision before any model call.

## Runtime

Validated on the isolated stormtrooper Docker host, using `poseidon-closeout-test:20260921` as the existing Python test base. The Dockerfile installs dsh `0.1.6-alpha.2`; mount the official Python SDK `python/sdk/src` from commit `ddefc45fbc7f8e46dd73185e68295696d1297887` at `/sdk`. npm transitive dependencies are resolved during build; retain the built image and version inventory for exact replay.

Mount Poseidon at `/workspace:ro`, this example at `/config:ro`, and a writable output directory at `/run-output`. Copy `input_bundle.json` to `/run-output/evidence/input_bundle.json`. Supply a dedicated mode-600 env file containing `OPENAI_API_KEY` through Docker `--env-file`; never put its value in command arguments or commit it. The profile routes directly to OpenAI and has no credential fallback.

Run these arguments inside the container, first with `2026-q1`, then `2026-q2`:

```sh
timeout 240 /app/.venv/bin/python /workspace/scripts/dsh_2330_poc.py \
  --bundle /run-output/evidence/input_bundle.json \
  --snapshot 2026-q1 --out-dir /run-output/revisions \
  --dsh-bin /usr/local/bin/dsh --profile sdk-minimal \
  --patches /config/research.patch.yml \
  --provider openai --model gpt-4.1-mini-2025-04-14
```

The profile exposes only `mcp__aquarium__read_snapshot` and `mcp__aquarium__read_previous_revision`. Filesystem, shell, subagent, and generic MCP resource tools are absent. Each model response is capped at 4,096 output tokens; provider retries are disabled. The 240-second outer timeout bounds the process. These are operational limits, not an aggregate dollar cap.

Outputs: `ledger.sqlite`, revision JSON/Markdown, and `raw-session-*.json` for successful or failed runtime results. Run one writer per output directory. Do not change profile content without also changing the runner policy version: idempotency does not fingerprint the profile or SDK automatically.

## Evidence and limits

- Financial facts were checked against page 1 of the official TSMC releases linked in the bundle. CDN downloads returned 403, so normalized facts and content hashes are retained; original PDF bytes/hashes are unavailable. Extraction is manually verified for this PoC, not a Triton integration.
- Each cutoff is the end of the disclosure date in Asia/Taipei. Exact intraday publication times are not verified. This is a research replay, not a trading backtest.
- Technical facts use stored raw closing prices, not adjusted prices or total returns. Adjustment factors showed discontinuities. The source database was retrieved later and has no point-in-time revision history here. Volume units are unverified; only a dimensionless ratio is used.
- Source-ID validation proves references exist and are date-eligible. It does not prove each sentence is entailed by its source. Human review remains necessary. Model pretraining can contain later knowledge despite prompt restrictions.
- Prices, financial facts, company guidance and AI inferences have different meanings. Company guidance is a dated forecast, not a realized result. This PoC does not produce price targets, orders or position sizing.

## Verification

Run on the designated remote test host:

```sh
python -m unittest tests.test_dsh_2330_poc -v
```

The initial real provider attempt on 2026-09-21 returned OpenAI `credit_balance_exhausted`; it produced no research revision. SDK/MCP initialization and a separate no-key request-header probe verified the two-tool surface. Live two-revision validation remains pending a funded model route.
