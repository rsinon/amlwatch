"""Detection rules.

Each rule takes a prepared transaction table and a :class:`RuleConfig` and
returns a list of :class:`Alert` objects, one per customer it flags. Rules are
deliberately transparent: every alert carries the transactions behind it and a
plain-language explanation an investigator can check.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import RuleConfig


@dataclass
class Alert:
    customer: str
    rule: str
    strength: float
    detail: str
    txn_ids: list[str] = field(default_factory=list)
    amount: float = 0.0
    first_seen: pd.Timestamp | None = None
    last_seen: pd.Timestamp | None = None


def _alert(customer, rule, strength, detail, rows: pd.DataFrame) -> Alert:
    return Alert(
        customer=customer,
        rule=rule,
        strength=float(np.clip(strength, 0.0, 1.0)),
        detail=detail,
        txn_ids=rows["txn_id"].tolist(),
        amount=float(rows["amount"].sum()),
        first_seen=rows["timestamp"].min(),
        last_seen=rows["timestamp"].max(),
    )


def _densest_window(times: np.ndarray, window: np.timedelta64) -> tuple[int, int]:
    """Return (start, stop) indices of the window holding the most events."""
    stops = np.searchsorted(times, times + window, side="right")
    sizes = stops - np.arange(times.size)
    i = int(np.argmax(sizes))
    return i, int(stops[i])


def structuring(df: pd.DataFrame, cfg: RuleConfig) -> list[Alert]:
    """Repeated cash deposits just below the reporting threshold."""
    lo = cfg.reporting_threshold * cfg.structuring_lower_ratio
    cand = df[
        (df["channel"] == "cash")
        & (df["sender"] == cfg.cash_account)
        & df["receiver_is_customer"]
        & (df["amount"] >= lo)
        & (df["amount"] < cfg.reporting_threshold)
    ]
    window = np.timedelta64(cfg.structuring_window_days, "D")
    alerts = []
    for cust, g in cand.groupby("receiver", sort=False):
        if len(g) < cfg.structuring_min_count:
            continue
        i, j = _densest_window(g["timestamp"].to_numpy(), window)
        count = j - i
        if count >= cfg.structuring_min_count:
            rows = g.iloc[i:j]
            strength = 0.5 + 0.5 * (count - cfg.structuring_min_count) / cfg.structuring_min_count
            alerts.append(_alert(
                cust, "structuring", strength,
                f"{count} cash deposits between {lo:,.0f} and {cfg.reporting_threshold:,.0f} "
                f"within {cfg.structuring_window_days} days",
                rows,
            ))
    return alerts


def pass_through(df: pd.DataFrame, cfg: RuleConfig) -> list[Alert]:
    """Large inflows forwarded on quickly at a similar value."""
    inflow = df[
        df["receiver_is_customer"]
        & (df["channel"] == "transfer")
        & (df["amount"] >= cfg.passthrough_min_amount)
    ][["txn_id", "timestamp", "receiver", "amount"]].rename(
        columns={"receiver": "customer", "txn_id": "in_id", "amount": "in_amount"})
    outflow = df[df["sender_is_customer"]][["txn_id", "timestamp", "sender", "amount"]].rename(
        columns={"sender": "customer", "txn_id": "out_id", "amount": "out_amount"})
    if inflow.empty or outflow.empty:
        return []
    outflow = outflow.rename(columns={"timestamp": "out_time"})
    outflow = outflow[outflow["customer"].isin(inflow["customer"].unique())]

    # Pair every large inflow with every later outflow from the same customer,
    # keep pairs inside the window at a similar value, then take the first
    # matching outflow per inflow, using each outflow at most once.
    pairs = inflow.merge(outflow, on="customer")
    lag = pairs["out_time"] - pairs["timestamp"]
    ratio = pairs["out_amount"] / pairs["in_amount"]
    pairs = pairs[
        (lag > pd.Timedelta(0))
        & (lag <= pd.Timedelta(hours=cfg.passthrough_window_hours))
        & (ratio >= cfg.passthrough_min_ratio)
        & (ratio <= 1.0)
    ].sort_values(["timestamp", "out_time"])
    pairs = pairs.drop_duplicates("in_id").drop_duplicates("out_id")

    large_inflows = inflow["customer"].value_counts()
    large_value = inflow.groupby("customer")["in_amount"].sum()
    by_id = df.set_index("txn_id")
    alerts = []
    for cust, g in pairs.groupby("customer", sort=False):
        episodes = len(g)
        share = g["in_amount"].sum() / large_value[cust]
        if episodes < cfg.passthrough_min_episodes or share < cfg.passthrough_min_share:
            continue
        ids = pd.unique(pd.concat([g["in_id"], g["out_id"]]))
        rows = by_id.loc[ids].reset_index()
        hours = ((g["out_time"] - g["timestamp"]).dt.total_seconds() / 3600).median()
        strength = 0.5 + 0.25 * (episodes - cfg.passthrough_min_episodes)
        alerts.append(_alert(
            cust, "pass_through", strength,
            f"{episodes} of {large_inflows[cust]} inflows of {cfg.passthrough_min_amount:,.0f}+ "
            f"({share:.0%} of that value) forwarded at {cfg.passthrough_min_ratio:.0%}+, "
            f"median {hours:.1f} hours later",
            rows,
        ))
    return alerts


def fan_in(df: pd.DataFrame, cfg: RuleConfig) -> list[Alert]:
    """Many distinct senders paying one customer small amounts in a short window."""
    cand = df[
        df["receiver_is_customer"]
        & (df["channel"] == "transfer")
        & (df["amount"] <= cfg.fanin_max_amount)
    ]
    window = pd.Timedelta(days=cfg.fanin_window_days)
    alerts = []
    for cust, g in cand.groupby("receiver", sort=False):
        if g["sender"].nunique() < cfg.fanin_min_senders:
            continue
        times = g["timestamp"].to_numpy()
        senders = g["sender"].to_numpy()
        seen: Counter = Counter()
        best, best_span, lo = 0, (0, 0), 0
        for hi in range(len(g)):
            seen[senders[hi]] += 1
            while times[hi] - times[lo] > window:
                seen[senders[lo]] -= 1
                if seen[senders[lo]] == 0:
                    del seen[senders[lo]]
                lo += 1
            if len(seen) > best:
                best, best_span = len(seen), (lo, hi + 1)
        if best >= cfg.fanin_min_senders:
            rows = g.iloc[best_span[0]:best_span[1]]
            strength = 0.5 + 0.5 * (best - cfg.fanin_min_senders) / cfg.fanin_min_senders
            alerts.append(_alert(
                cust, "fan_in", strength,
                f"{best} distinct senders of amounts up to {cfg.fanin_max_amount:,.0f} "
                f"within {cfg.fanin_window_days} days",
                rows,
            ))
    return alerts


def high_risk_corridor(df: pd.DataFrame, cfg: RuleConfig) -> list[Alert]:
    """Transfers to or from listed high-risk jurisdictions."""
    if not cfg.high_risk_countries:
        return []
    hr = set(cfg.high_risk_countries)
    big = (df["channel"] == "transfer") & (df["amount"] >= cfg.high_risk_min_amount)
    out_ = df[big & df["sender_is_customer"] & df["receiver_country"].isin(hr)]
    in_ = df[big & df["receiver_is_customer"] & df["sender_country"].isin(hr)]
    hits = pd.concat([out_.assign(customer=out_["sender"]), in_.assign(customer=in_["receiver"])])
    alerts = []
    for cust, g in hits.groupby("customer", sort=False):
        if len(g) < cfg.high_risk_min_count:
            continue
        countries = ", ".join(sorted((set(g["receiver_country"]) | set(g["sender_country"])) & hr))
        strength = 0.3 + 0.1 * len(g)
        alerts.append(_alert(
            cust, "high_risk_corridor", strength,
            f"{len(g)} transfers totalling {g['amount'].sum():,.0f} with {countries}",
            g,
        ))
    return alerts


def round_trip(df: pd.DataFrame, cfg: RuleConfig) -> list[Alert]:
    """Value that leaves a customer and returns through a chain of customers.

    Searches for time-ordered chains A -> B -> ... -> A where each hop happens
    after the previous one, within the window, and keeps most of the value.
    """
    edges = df[
        df["sender_is_customer"]
        & df["receiver_is_customer"]
        & (df["channel"] == "transfer")
        & (df["amount"] >= cfg.roundtrip_min_amount)
    ]
    if edges.empty:
        return []
    window = np.timedelta64(cfg.roundtrip_window_days, "D")
    ids = edges["txn_id"].to_numpy()
    snd = edges["sender"].to_numpy()
    rcv = edges["receiver"].to_numpy()
    amt = edges["amount"].to_numpy()
    tms = edges["timestamp"].to_numpy()

    outgoing: dict[str, list[int]] = defaultdict(list)
    for k in range(len(edges)):
        outgoing[snd[k]].append(k)  # already in time order

    found: dict[str, set[int]] = defaultdict(set)
    cycles: dict[str, int] = defaultdict(int)

    def extend(path: list[int], origin: str, deadline) -> None:
        last = path[-1]
        node = rcv[last]
        if node == origin and len(path) >= 2:
            members = {snd[p] for p in path}
            for m in members:
                found[m].update(path)
            cycles[origin] += 1
            return
        if len(path) >= cfg.roundtrip_max_length:
            return
        visited = {snd[p] for p in path}
        for nxt in outgoing.get(node, ()):
            if tms[nxt] <= tms[last]:
                continue
            if tms[nxt] > deadline:
                break
            if rcv[nxt] in visited and rcv[nxt] != origin:
                continue
            ratio = amt[nxt] / amt[last]
            if cfg.roundtrip_min_ratio <= ratio <= 1.0:
                extend(path + [nxt], origin, deadline)

    for k in range(len(edges)):
        extend([k], snd[k], tms[k] + window)

    alerts = []
    by_id = edges.set_index("txn_id")
    for cust, idx in found.items():
        rows = by_id.loc[ids[sorted(idx)]].reset_index()
        n_cycles = max(cycles.get(cust, 0), 1)
        strength = 0.6 + 0.1 * n_cycles
        alerts.append(_alert(
            cust, "round_trip", strength,
            f"part of circular flows returning {cfg.roundtrip_min_ratio:.0%}+ of value "
            f"within {cfg.roundtrip_window_days} days ({len(rows)} linked transfers)",
            rows,
        ))
    return alerts


RULES: dict[str, Callable[[pd.DataFrame, RuleConfig], list[Alert]]] = {
    "structuring": structuring,
    "pass_through": pass_through,
    "fan_in": fan_in,
    "high_risk_corridor": high_risk_corridor,
    "round_trip": round_trip,
}
