"""Configuration for the monitoring rules and scoring."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

DEFAULT_WEIGHTS: dict[str, float] = {
    "structuring": 0.6,
    "pass_through": 0.5,
    "fan_in": 0.5,
    "high_risk_corridor": 0.4,
    "round_trip": 0.7,
}


@dataclass
class RuleConfig:
    """Thresholds for every detection rule.

    Defaults are illustrative. Real deployments should calibrate them
    against their own customer base and regulatory thresholds.
    """

    # Accounts whose identifier matches this regex are treated as customers
    # (the institution's own accounts). Ignored when the input already has
    # ``sender_is_customer`` / ``receiver_is_customer`` columns.
    customer_pattern: str = r"^C\d+$"
    cash_account: str = "CASH"

    # Structuring: repeated cash deposits just under the reporting threshold
    reporting_threshold: float = 10_000.0
    structuring_lower_ratio: float = 0.8
    structuring_window_days: int = 7
    structuring_min_count: int = 3

    # Pass-through: funds received and sent on quickly at a similar value
    passthrough_min_amount: float = 3_000.0
    passthrough_window_hours: float = 48.0
    passthrough_min_ratio: float = 0.9
    passthrough_min_episodes: int = 3
    # Share of the value of the customer's large inflows that was forwarded on
    passthrough_min_share: float = 0.5

    # Fan-in: many distinct senders paying one account in a short window
    fanin_window_days: int = 7
    fanin_min_senders: int = 12
    fanin_max_amount: float = 3_000.0

    # High-risk corridor: transfers to or from listed jurisdictions.
    # The defaults are placeholder codes; supply your own list.
    high_risk_countries: tuple[str, ...] = ("ZZ", "ZY")
    high_risk_min_amount: float = 1_000.0
    high_risk_min_count: int = 2

    # Round-trip: value that leaves an account and returns through a chain
    roundtrip_min_amount: float = 2_000.0
    roundtrip_window_days: int = 14
    roundtrip_max_length: int = 4
    roundtrip_min_ratio: float = 0.85

    # Scoring
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    anomaly_weight: float = 0.3
    anomaly_review_quantile: float = 0.99
    random_state: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuleConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        data = dict(data)
        if "high_risk_countries" in data:
            data["high_risk_countries"] = tuple(data["high_risk_countries"])
        if "weights" in data:
            data["weights"] = {**DEFAULT_WEIGHTS, **data["weights"]}
        return cls(**data)

    @classmethod
    def from_toml(cls, path: str | Path) -> RuleConfig:
        with open(path, "rb") as fh:
            return cls.from_dict(tomllib.load(fh))
