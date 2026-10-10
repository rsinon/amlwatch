"""Measure detection quality against labelled typologies."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .monitor import MonitoringResult


def labelled_customers(df: pd.DataFrame) -> pd.Series:
    """Map each customer involved in a labelled typology to that typology."""
    lab = df[df["typology"].astype(str) != "none"]
    parts = [
        lab.loc[lab["sender_is_customer"], ["sender", "typology"]].rename(columns={"sender": "customer"}),
        lab.loc[lab["receiver_is_customer"], ["receiver", "typology"]].rename(columns={"receiver": "customer"}),
    ]
    both = pd.concat(parts).drop_duplicates("customer")
    return both.set_index("customer")["typology"]


def _row(name: str, flagged: set, positives: set) -> dict:
    tp = len(flagged & positives)
    precision = tp / len(flagged) if flagged else float("nan")
    recall = tp / len(positives) if positives else float("nan")
    valid = not (math.isnan(precision) or math.isnan(recall)) and (precision + recall) > 0
    f1 = 2 * precision * recall / (precision + recall) if valid else float("nan")
    return {"scope": name, "flagged": len(flagged), "positives": len(positives),
            "true_positives": tp, "precision": precision, "recall": recall, "f1": f1}


def evaluate(result: MonitoringResult) -> pd.DataFrame | None:
    df = result.transactions
    if "typology" not in df.columns:
        return None
    truth = labelled_customers(df)
    positives = set(truth.index)
    alerted = set(result.customers.index[result.customers["status"] == "alert"])

    nan = float("nan")
    rows = [_row("all alerts", alerted, positives)]

    # Recall within each typology: what share of its customers were alerted.
    for typ in sorted(truth.unique()):
        members = set(truth.index[truth == typ])
        hit = len(alerted & members)
        rows.append({"scope": f"typology: {typ}", "flagged": hit, "positives": len(members),
                     "true_positives": hit, "precision": nan,
                     "recall": hit / len(members), "f1": nan})

    # Precision of each rule: what share of its alerts were real typology customers.
    if len(result.alerts):
        for rule, g in result.alerts.groupby("rule"):
            r = _row(f"rule: {rule}", set(g["customer"]), positives)
            r.update(positives=nan, recall=nan, f1=nan)
            rows.append(r)
    return pd.DataFrame(rows).set_index("scope")
