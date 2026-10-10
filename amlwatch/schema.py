"""Input validation and normalisation."""

from __future__ import annotations

import pandas as pd

from .config import RuleConfig

REQUIRED = ("txn_id", "timestamp", "sender", "receiver", "amount", "channel")
OPTIONAL_DEFAULTS = {"sender_country": "", "receiver_country": "", "currency": ""}


class SchemaError(ValueError):
    """Raised when a transaction table is missing required columns or values."""


def prepare(df: pd.DataFrame, config: RuleConfig) -> pd.DataFrame:
    """Validate a transaction table and return a normalised copy.

    The returned frame is sorted by time and always has the boolean columns
    ``sender_is_customer`` and ``receiver_is_customer``.
    """
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")

    out = df.copy()
    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in out.columns:
            out[col] = default
        out[col] = out[col].fillna(default).astype(str)

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    bad = out["timestamp"].isna() | out["amount"].isna()
    if bad.any():
        raise SchemaError(f"{int(bad.sum())} rows have an unreadable timestamp or amount")
    if (out["amount"] < 0).any():
        raise SchemaError("Amounts must be non-negative; record direction with sender/receiver")

    for col in ("txn_id", "sender", "receiver", "channel"):
        out[col] = out[col].astype(str)
    out["channel"] = out["channel"].str.lower()

    if not {"sender_is_customer", "receiver_is_customer"} <= set(out.columns):
        pattern = config.customer_pattern
        out["sender_is_customer"] = out["sender"].str.match(pattern)
        out["receiver_is_customer"] = out["receiver"].str.match(pattern)
    out["sender_is_customer"] = out["sender_is_customer"].astype(bool)
    out["receiver_is_customer"] = out["receiver_is_customer"].astype(bool)

    return out.sort_values("timestamp", kind="stable").reset_index(drop=True)
