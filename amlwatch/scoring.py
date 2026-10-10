"""Customer-level behavioural features and anomaly scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .config import RuleConfig

FEATURES = [
    "n_out",
    "n_in",
    "value_out",
    "value_in",
    "cash_in_share",
    "cash_out_share",
    "distinct_senders",
    "distinct_receivers",
    "foreign_share",
    "max_amount",
    "flow_through",
    "night_share",
]


def customer_features(df: pd.DataFrame, cfg: RuleConfig) -> pd.DataFrame:
    """Summarise each customer's activity into one row of features."""
    home = _home(df)
    night = (df["timestamp"].dt.hour < 6) | (df["timestamp"].dt.hour >= 22)
    cash_amt = df["amount"].where(df["channel"] == "cash", 0.0)
    base = df.assign(night=night.astype(int), cash_amt=cash_amt)

    out = base[base["sender_is_customer"]].assign(
        customer=lambda d: d["sender"],
        foreign=lambda d: (~d["receiver_country"].isin(["", home])).astype(int))
    inc = base[base["receiver_is_customer"]].assign(
        customer=lambda d: d["receiver"],
        foreign=lambda d: (~d["sender_country"].isin(["", home])).astype(int))

    o = out.groupby("customer").agg(
        n_out=("txn_id", "size"),
        value_out=("amount", "sum"),
        max_out=("amount", "max"),
        distinct_receivers=("receiver", "nunique"),
        cash_out_value=("cash_amt", "sum"),
        foreign_out=("foreign", "sum"),
        night_out=("night", "sum"),
    )
    i = inc.groupby("customer").agg(
        n_in=("txn_id", "size"),
        value_in=("amount", "sum"),
        max_in=("amount", "max"),
        distinct_senders=("sender", "nunique"),
        cash_in_value=("cash_amt", "sum"),
        foreign_in=("foreign", "sum"),
        night_in=("night", "sum"),
    )
    f = o.join(i, how="outer").fillna(0.0)
    n_total = (f["n_out"] + f["n_in"]).replace(0, np.nan)
    f["cash_in_share"] = f["cash_in_value"] / f["value_in"].replace(0, np.nan)
    f["cash_out_share"] = f["cash_out_value"] / f["value_out"].replace(0, np.nan)
    f["foreign_share"] = (f["foreign_out"] + f["foreign_in"]) / n_total
    f["night_share"] = (f["night_out"] + f["night_in"]) / n_total
    f["max_amount"] = f[["max_out", "max_in"]].max(axis=1)
    f["flow_through"] = np.minimum(f["value_out"], f["value_in"]) / np.maximum(
        f[["value_out", "value_in"]].max(axis=1), 1.0)
    f = f.fillna(0.0)
    f.index.name = "customer"
    return f[FEATURES]


def _home(df: pd.DataFrame) -> str:
    """The most common customer-side country, treated as the home market."""
    sc = df.loc[df["sender_is_customer"], "sender_country"]
    mode = sc[sc != ""].mode()
    return str(mode.iloc[0]) if len(mode) else ""


def anomaly_scores(features: pd.DataFrame, cfg: RuleConfig) -> pd.Series:
    """Isolation-forest anomaly score per customer, as a percentile in [0, 1].

    Higher means more unusual relative to the rest of the book.
    """
    if len(features) < 10:
        return pd.Series(0.0, index=features.index, name="anomaly")
    x = features.copy()
    for col in ("n_out", "n_in", "value_out", "value_in", "distinct_senders",
                "distinct_receivers", "max_amount"):
        x[col] = np.log1p(x[col])
    x = StandardScaler().fit_transform(x)
    model = IsolationForest(n_estimators=300, random_state=cfg.random_state)
    model.fit(x)
    raw = -model.score_samples(x)
    return pd.Series(raw, index=features.index).rank(pct=True).rename("anomaly")
