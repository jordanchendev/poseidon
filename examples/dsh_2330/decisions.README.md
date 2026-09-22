# Bounded decision replacement

`dsh_decision_poc.py` asks one provider question packet per snapshot. It replaces only the evidence-relevance, revision-route, and sufficiency decisions; it does not generate research prose or an investment conclusion. Choice probabilities express the model's choice distribution, not probabilities of market outcomes.

The packet JSON is a list with `id`, `snapshot_id`, `snapshot_digest`, `state`, `questions`, and optional `evidence_question_ids`. Every question is a closed `choice`; the runner rejects missing options, malformed probabilities, non-argmax choices, an unpinned returned model, or invalid token usage. It makes no cache lookup and no retry.

Run a fresh measured JEV arm in the isolated container with a dedicated mode-600 env file:

```sh
/app/.venv/bin/python /workspace/scripts/dsh_decision_poc.py \
  --provider jev --packets /run-output/evidence/packets.json \
  --out-dir /run-output/jev-decisions --repeats 3
```

The OpenAI arm uses the same `state` and `questions` in a JSON-output Chat Completions prompt:

```sh
/app/.venv/bin/python /workspace/scripts/dsh_decision_poc.py \
  --provider openai --packets /run-output/evidence/packets.json \
  --out-dir /run-output/openai-decisions --repeats 3
```

Each new output directory contains one request/response record per packet/repeat plus `records.json`; existing names are rejected before any request. Records retain raw successful request/response payloads, elapsed API time, returned model, normalized token usage (including cached input tokens separately), HTTP status/request ID when supplied, and failures without raw HTTP bodies. A failed request is never a successful zero-cost measurement.

`dsh_2330_poc.py --decision-artifact ARTIFACT.json` is optional. The artifact has `{"packet": <full frozen packet>, "records": <raw records list>}`. The loader recomputes answers from each raw provider response and the full packet; it does not trust saved decision fields. It verifies packet/snapshot/context digests, requires all selected repeats to agree, fails closed on review or insufficient decisions, filters the generator's evidence to host-approved IDs, disables the old aquarium MCP surface, and adds decision provider/model/policy/context provenance to the revision identity. `create` requires no prior revision, while `update` and `no_change` require one; `no_change` returns that retained revision without narrative generation. Omitting the flag preserves the legacy runner.

## Default JEV research flow

`dsh_research.py` builds a cutoff-filtered packet, finds the actual predecessor
in the SQLite ledger, and uses JEV by default. It has no OpenAI or narrative
API fallback. Supply `JEV_DSH_ANALYSIS_API_KEY` through the runtime environment;
do not put its value in command arguments or request/reply files. An orchestrator
passes exported requests to its narrative/review subagent; the CLI does not
launch a Codex agent itself. Run Q1 first and use the same ledger for Q2:

```sh
/app/.venv/bin/python /workspace/scripts/dsh_research.py prepare \
  --bundle /run-output/evidence.json --snapshot 2026-q1 \
  --ledger /run-output/jev-research/ledger.sqlite \
  --request /run-output/jev-research/q1-request.json
```

The JSON result is either `generation`, `review`, or `no_change`. A generation
request includes `request_digest`, the selected snapshot, a bounded prose
prompt, and JEV provenance. Terra replies with that exact digest, its actual
`provider` and `model`, and a `research` object matching the existing narrative
contract. Persist it with:

```sh
/app/.venv/bin/python /workspace/scripts/dsh_research.py complete \
  --bundle /run-output/evidence.json \
  --ledger /run-output/jev-research/ledger.sqlite \
  --request /run-output/jev-research/q1-request.json \
  --reply /run-output/jev-research/q1-reply.json \
  --out-dir /run-output/jev-research/reports
```

`review` is emitted for provider errors, malformed answers, uncertainty,
conflict, missing required fundamental/technical evidence, insufficient
research, invalid route, or a selected choice probability below the operational
`0.8` policy threshold. The threshold is a workflow guard, not a calibrated
market probability. Valuation insufficiency remains a narrative limitation and
does not alone block bounded research.

To resolve review, the reply must also include `"resolution":"resolved"` and
a full valid `answers` object for every question in `decision_packet`. The
workflow revalidates those answers and all gates; unresolved review cannot
create a revision. `no_change` returns the actual prior revision and needs no
narrative. Repeated prepare reuses its saved JEV attempt; repeated complete
returns the immutable revision.

A generation reply has this envelope:

```json
{"request_digest":"<exact digest from request>","provider":"codex-subagent","model":"gpt-5.6-terra","research":{}}
```

Replace `research` with the object required by the exported prompt: `symbol`,
`as_of`, `thesis`, `stance`, `claims`, `risks`, `invalidation_conditions`,
`unknowns`, `change_summary`, and `next_review_at`. Each claim needs `kind`,
`text`, and `evidence_ids`. A resolved review may instead select `no_change`:
then omit `research`; the existing predecessor and review provenance are returned
without generating a new report. Requests are bound to the ledger's canonical
copy; modifying a request and recomputing its digest does not authorize it.
