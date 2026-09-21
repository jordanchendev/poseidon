"""Qlib record template for the canonical v18 basis rule."""

from __future__ import annotations


class BasisRuleRecord:
    """Persist the v18 B-rule return series in the qrun recorder."""

    artifact_path = "basis_rule"

    def __init__(self, recorder, start_time: str, end_time: str, warmup: int = 252):
        self._recorder = recorder
        self.start_time = start_time
        self.end_time = end_time
        self.warmup = warmup

    @property
    def recorder(self):
        return self._recorder

    def generate(self, **kwargs):
        import pandas as pd
        from qlib.data import D

        from poseidon.research.tx_basis_rule import canonical_returns

        data = D.features(
            ["TX", "0050"],
            fields=["$open", "$close"],
            start_time=pd.Timestamp(self.start_time),
            end_time=pd.Timestamp(self.end_time),
        )
        tx = data.xs("TX", level="instrument").rename(columns={"$open": "open", "$close": "close"})
        tw0050 = data.xs("0050", level="instrument").rename(columns={"$open": "open", "$close": "close"})
        net, engaged, metrics = canonical_returns(tx, tw0050, warmup=self.warmup)
        self.recorder.save_objects(
            artifact_path=self.artifact_path,
            **{"returns.pkl": net, "engaged.pkl": engaged, "metrics.pkl": metrics},
        )

    def list(self):
        return ["returns.pkl", "engaged.pkl", "metrics.pkl"]
