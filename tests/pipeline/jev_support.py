"""A fake OpenRouter (``GET /key`` + ``POST /systemone``) for JEV tests, served through ``httpx.MockTransport``.

The real ``JevClient`` / ``OpenRouterClient`` code path runs; only the network is replaced.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx

from poseidon.pipeline.faithfulness import SpecFaithfulnessStage
from poseidon.pipeline.jev import transport_client_factory
from poseidon.pipeline.openrouter import KEY_ENV

KEY = "test-openrouter-key-not-real"
LABEL = "sk-or-v1-fee...567"
ENV = {KEY_ENV: KEY}
COST = 0.00004

# (question name, state, criteria) -> choice, or (choice, probability) to set the reply's probability
Rule = Callable[[str, dict, dict], str | tuple[str, float]]


def default_rule(name: str, state: dict, criteria: dict) -> str:
    """Everything is a supported leverage clause, unless a keyword says otherwise."""

    clause = state.get("clause", "")
    if name == "clause_slot":
        for word, slot in (("put", "hedge"), ("換月", "roll"), ("微台", "instrument"), ("你們", "not_strategy")):
            if word in clause:
                return slot
        return "leverage"
    if name == "clause_component":
        if "均線" in clause and "regime.ma_band" in criteria:
            return "regime.ma_band"
        if "波動" in clause and "risk.vol_cap" in criteria:
            return "risk.vol_cap"
        return next(iter(criteria))
    if "put" in clause or "你們" in clause:
        return "says_nothing"
    return "supports"


class FakeOpenRouter:
    def __init__(
        self,
        rule: Rule = default_rule,
        *,
        label: str = LABEL,
        limit_remaining: float = 19.5,
        cost: float | None = COST,
        systemone_status: int = 200,
        error_message: str = "",
    ):
        self.rule = rule
        self.label = label
        self.limit_remaining = limit_remaining
        self.cost = cost
        self.systemone_status = systemone_status
        self.error_message = error_message
        self.requests: list[httpx.Request] = []

    @property
    def systemone_bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.url.path.endswith("/systemone")]

    @property
    def key_checks(self) -> int:
        return sum(1 for r in self.requests if r.url.path.endswith("/key"))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/api/v1/key":
            data = {"label": self.label, "limit": 20, "limit_remaining": self.limit_remaining, "usage": 0.5}
            return httpx.Response(200, json={"data": data})
        assert request.method == "POST" and request.url.path == "/api/v1/systemone", request.url
        if self.systemone_status != 200:
            return httpx.Response(self.systemone_status, json={"error": {"message": self.error_message}})
        body = json.loads(request.content)
        ((name, question),) = body["questions"].items()
        choice = self.rule(name, body["state"], question["criteria"])
        choice, p = choice if isinstance(choice, tuple) else (choice, 0.9)
        usage = {"input_tokens": 800, "output_tokens": 40}
        if self.cost is not None:
            usage["cost"] = self.cost
        return httpx.Response(
            200,
            json={
                "model": "typesafe/jev-1.13-20260917",
                "answers": {name: {"type": "choice", "choice": choice, "probabilities": {choice: p}, "confidence": p}},
                "usage": usage,
                "provider": "TypeSafe",
            },
        )

    def factory(self):
        return transport_client_factory(httpx.MockTransport(self))


def fake_faithfulness_stage(fake: FakeOpenRouter | None = None, *, env=None, pin=LABEL) -> SpecFaithfulnessStage:
    fake = fake or FakeOpenRouter()
    return SpecFaithfulnessStage(
        client_factory=fake.factory(), pin_provider=lambda: pin, environ=ENV if env is None else env
    )
