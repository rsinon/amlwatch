"""The monitoring pipeline: rules, scoring, triage and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RuleConfig
from .rules import RULES, Alert
from .schema import prepare
from .scoring import anomaly_scores, customer_features


@dataclass
class MonitoringResult:
    """Output of a monitoring run.

    Attributes
    ----------
    alerts:
        One row per (customer, rule) alert, with the evidence behind it.
    customers:
        One row per customer with features, rule hits, anomaly percentile,
        a combined risk score (0-100) and a triage status.
    transactions:
        The prepared input table.
    config:
        The configuration used.
    """

    alerts: pd.DataFrame
    customers: pd.DataFrame
    transactions: pd.DataFrame
    config: RuleConfig

    @property
    def alerted(self) -> pd.DataFrame:
        """Customers with at least one rule alert, highest risk first."""
        c = self.customers
        return c[c["status"] == "alert"].sort_values("risk_score", ascending=False)

    def summary(self) -> dict:
        c = self.customers
        return {
            "transactions": len(self.transactions),
            "customers": len(c),
            "alerted_customers": int((c["status"] == "alert").sum()),
            "anomaly_review": int((c["status"] == "anomaly_review").sum()),
            "alerts_by_rule": self.alerts["rule"].value_counts().to_dict() if len(self.alerts) else {},
        }

    def evaluate(self) -> pd.DataFrame | None:
        """Score alerts against a ``typology`` label column, if present.

        Returns a table with precision and recall overall and per typology,
        or ``None`` when the input has no labels.
        """
        from .evaluate import evaluate

        return evaluate(self)

    def to_html(self, path: str | Path, top: int = 50) -> Path:
        from .report import render_html

        path = Path(path)
        path.write_text(render_html(self, top=top), encoding="utf-8")
        return path


class Monitor:
    """Run transaction monitoring over a table of transactions.

    Example
    -------
    >>> from amlwatch import Monitor, generate_transactions
    >>> result = Monitor().run(generate_transactions(n_customers=300, seed=7))
    >>> result.alerted.head()  # doctest: +SKIP
    """

    def __init__(self, config: RuleConfig | None = None, rules: list[str] | None = None):
        self.config = config or RuleConfig()
        self.rules = rules or list(RULES)
        unknown = set(self.rules) - set(RULES)
        if unknown:
            raise ValueError(f"Unknown rules: {sorted(unknown)}")

    def run(self, transactions: pd.DataFrame) -> MonitoringResult:
        cfg = self.config
        df = prepare(transactions, cfg)

        found: list[Alert] = []
        for name in self.rules:
            found.extend(RULES[name](df, cfg))
        alerts = pd.DataFrame([asdict(a) for a in found],
                              columns=list(Alert.__dataclass_fields__))
        if len(alerts):
            alerts["n_txns"] = alerts["txn_ids"].str.len()
            alerts["weighted"] = alerts.apply(
                lambda r: cfg.weights.get(r["rule"], 0.5) * r["strength"], axis=1)
            alerts = alerts.sort_values(["customer", "rule"]).reset_index(drop=True)
        else:
            alerts["n_txns"] = pd.Series(dtype=int)
            alerts["weighted"] = pd.Series(dtype=float)

        feats = customer_features(df, cfg)
        anomaly = anomaly_scores(feats, cfg)
        customers = feats.join(anomaly)

        if len(alerts):
            # Combine rule evidence with a noisy-OR so independent rules reinforce
            # each other without the total exceeding 1.
            rule_score = alerts.groupby("customer")["weighted"].apply(
                lambda w: 1 - np.prod(1 - w.clip(0, 0.99)))
            rules_hit = alerts.groupby("customer")["rule"].apply(lambda r: ", ".join(sorted(r)))
        else:
            rule_score = pd.Series(dtype=float)
            rules_hit = pd.Series(dtype=str)

        customers["rule_score"] = rule_score.reindex(customers.index).fillna(0.0)
        customers["rules"] = rules_hit.reindex(customers.index).fillna("")
        w = cfg.anomaly_weight
        customers["risk_score"] = (
            100 * ((1 - w) * customers["rule_score"] + w * customers["anomaly"])
        ).round(1)

        review_cut = customers["anomaly"].quantile(cfg.anomaly_review_quantile)
        customers["status"] = np.select(
            [customers["rules"] != "", customers["anomaly"] >= review_cut],
            ["alert", "anomaly_review"],
            default="clear",
        )
        customers = customers.sort_values("risk_score", ascending=False)
        return MonitoringResult(alerts=alerts, customers=customers, transactions=df, config=cfg)
