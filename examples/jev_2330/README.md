# 2330 JEV evidence-verification PoC

This experiment replaces the bounded claim-review step with JEV typed decisions. It reuses the dated financial/price snapshots from `../dsh_2330/input_bundle.json`. It does not generate research prose or claim to complete the blocked dsh/OpenAI run.

`cases.json` contains controlled claims written for this experiment, not output from dsh. Each snapshot has six model-evaluated cases (supported financial fact, contradicted financial fact, unsupported valuation claim, supported raw-price fact, and English counterparts for the Chinese contradiction/valuation cases) and one fabricated quote rejected by Python. Expected labels are evaluation metadata and must never be sent to the model.

The host checks source eligibility, matches any quote against the normalized source text, sends only that case and its source context to JEV, validates the structured response, then stores the result and renders a Chinese report. Source text is a manually checked normalized excerpt, not archived PDF bytes. Financial/price data limitations are documented in the dsh example.

## Model and scope

Direct API: `https://api.typesafe.ai/v1/systemone`, model `jev-1.13.0`. Authentication uses only `JEV_DSH_ANALYSIS_API_KEY`; there is no generic-key fallback. The variable is supplied through a mode-600 Docker env file on the isolated test host. Never log authorization headers or commit credentials.

Use the existing Python test image; no new SDK dependency is required. Mount scripts at `/workspace/scripts`, example inputs at `/config`, output at `/run-output`, and put `/workspace/scripts` on `PYTHONPATH`.

```sh
python /workspace/scripts/jev_2330_poc.py \
  --bundle /config/input_bundle.json --cases /config/cases.json \
  --snapshot 2026-q1 --out-dir /run-output
```

Run Q1 then Q2 in the same output directory. Replaying a request reuses its stored result before credentials or network access. The experiment uses one writer and bounded requests; it has no background scheduler or trading integration.

`confidence >= 0.8` is a provisional routing threshold for this experiment. Confidence summarizes the model's distribution; it is not a measured probability that a financial conclusion is correct, nor the probability of investment profit. Small controlled cases establish API/flow feasibility, not domain-wide accuracy or calibration. Arithmetic, dates, version links and repeat suppression remain deterministic host responsibilities.

Published input pricing is $0.042 per million tokens, output free. Reported token-based costs are estimates at that rate, not invoice measurements. Model and price source: [Models](https://docs.typesafe.ai/models).

Official design references: [API](https://docs.typesafe.ai/api), [citation checking](https://docs.typesafe.ai/cookbooks/citation_check), [confidence](https://docs.typesafe.ai/confidence), [closed-set function dispatch](https://docs.typesafe.ai/cookbooks/function_calling).

## Observed run — 2026-09-21

The isolated live experiment completed 12 `jev-1.13.0` calls. All 12 controlled model cases matched their expected labels; Python rejected both fabricated quotes before an API request. Chinese/English contradiction and unsupported-valuation pairs agreed. The median observed HTTP call latency was 0.631 seconds (range 0.524–0.718 seconds).

The API reported 9,463 input tokens and 564 output tokens: estimated $0.000397446 at the published rate. Two immutable snapshot revisions were stored, Q2 linked to Q1. Both replayed successfully in a Docker container with networking disabled and no key; the SQLite dump and all case artifact hashes stayed unchanged. This proves the exercised replay path, not concurrency safety or calibrated financial accuracy.

Remote offline verification: 10 tests passed across the dsh and JEV PoC suites, including response probability checks, cutoff/citation restrictions, immutable revision linkage and replay without model access. The original dsh research run remains blocked by exhausted OpenAI credits; these JEV claims are independently prepared controls, not generated dsh conclusions.
