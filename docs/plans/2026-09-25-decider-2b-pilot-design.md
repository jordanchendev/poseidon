# Decider 2B Pilot Design

## Purpose

This pilot measures a local Decider 2B service against frozen DSH decision
packets. It does not change the production DSH or JEV route. The pilot keeps
the current low-confidence, conflict, and insufficient-evidence review rules.

## Design

Stormtrooper runs `Mapika/decider-2b` in a temporary CUDA container. The
service binds to loopback only. The service accepts the TypeSafe-compatible
`POST /v1/systemone` request. The container has no public port and no API key.

The pilot sends saved DSH packets to Decider. It stores each request, response,
elapsed time, model revision, and verdict in a new output directory. It then
compares Decider answers with the retained JEV answers. The comparison includes
the create, update, no-change, irrelevant-source, missing-technical, and
conflicting-source cases.

The pilot preserves the host decision rules. An invalid response, a low
confidence answer, a conflict, or missing required evidence produces `review`.
The pilot never creates a research revision. Terra does not run during this
pilot because the question is decision safety and service cost.

## Acceptance

The pilot fails if any existing negative case becomes an automatic generation.
The pilot fails if Decider p95 service latency exceeds twice the recorded JEV
baseline. The result also records decision agreement and the review rate. A
passing pilot is evidence for a DSH provider adapter. It is not evidence for a
production replacement.

## Verification

Make sure that the container sees CUDA before it receives a packet. Run the
frozen fixtures twice. Make sure that both runs have the same verdicts. Save a
single JSON summary and the raw response records. Stop and remove the temporary
container after the pilot.
