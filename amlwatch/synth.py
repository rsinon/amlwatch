"""Synthetic transaction data with labelled money-laundering typologies.

The generator produces a realistic-looking ledger of customer activity and
injects known typologies into a small share of customers. Every injected
transaction carries its typology label, so detection rules can be measured
against ground truth. No real customer data is used or needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd

TYPOLOGIES: tuple[str, ...] = (
    "structuring",
    "pass_through",
    "fan_in",
    "round_trip",
    "high_risk_corridor",
)

COLUMNS = [
    "txn_id",
    "timestamp",
    "sender",
    "receiver",
    "amount",
    "currency",
    "channel",
    "sender_country",
    "receiver_country",
    "typology",
]

_FOREIGN = ("GB", "FR", "IN", "AE", "ZA", "MU", "KE", "DE")


@dataclass
class SynthConfig:
    n_customers: int = 1_000
    days: int = 90
    start: str = "2026-01-01"
    typology_share: float = 0.03
    reporting_threshold: float = 10_000.0
    home_country: str = "SC"
    currency: str = "USD"
    high_risk_countries: tuple[str, ...] = ("ZZ", "ZY")
    n_external: int = 3_000
    seed: int | None = 42


class _Builder:
    """Collects transaction rows as column lists before building a frame."""

    def __init__(self) -> None:
        self.cols: dict[str, list] = {c: [] for c in COLUMNS if c not in ("txn_id", "currency")}

    def add(self, ts, sender, receiver, amount, channel, s_country, r_country, typology):
        c = self.cols
        c["timestamp"].append(ts)
        c["sender"].append(sender)
        c["receiver"].append(receiver)
        c["amount"].append(round(float(amount), 2))
        c["channel"].append(channel)
        c["sender_country"].append(s_country)
        c["receiver_country"].append(r_country)
        c["typology"].append(typology)


def generate_transactions(config: SynthConfig | None = None, **overrides) -> pd.DataFrame:
    """Generate a labelled synthetic transaction ledger.

    Parameters
    ----------
    config:
        A :class:`SynthConfig`. Keyword overrides are applied on top of it,
        so ``generate_transactions(n_customers=200, seed=1)`` also works.

    Returns
    -------
    pandas.DataFrame
        One row per transaction, sorted by time. ``typology`` is ``"none"``
        for ordinary activity and names the typology otherwise.
    """
    cfg = config or SynthConfig()
    if overrides:
        cfg = SynthConfig(**{**cfg.__dict__, **overrides})
    if cfg.n_customers < 10:
        raise ValueError("n_customers must be at least 10")

    rng = np.random.default_rng(cfg.seed)
    start = pd.Timestamp(cfg.start)
    home = cfg.home_country

    n = cfg.n_customers
    customers = np.array([f"C{i:05d}" for i in range(n)])
    is_business = rng.random(n) < 0.15
    base = np.where(
        is_business,
        rng.lognormal(np.log(2_500), 0.6, n),
        rng.lognormal(np.log(150), 0.6, n),
    )
    rate = np.where(is_business, 2.0, 0.4)

    externals = np.array([f"X{i:05d}" for i in range(cfg.n_external)])
    ext_country = np.where(
        rng.random(cfg.n_external) < 0.9,
        home,
        rng.choice(_FOREIGN, cfg.n_external),
    ).astype(object)
    # A handful of legitimate counterparties in listed jurisdictions, so the
    # corridor rule faces realistic noise rather than a perfectly clean signal.
    legit_hr = rng.random(cfg.n_external) < 0.003
    if cfg.high_risk_countries:
        ext_country[legit_hr] = rng.choice(cfg.high_risk_countries, legit_hr.sum())

    # ---- ordinary activity (vectorised) -------------------------------
    counts = rng.poisson(rate * cfg.days)
    owner = np.repeat(np.arange(n), counts)
    m = owner.size
    day = rng.integers(0, cfg.days, m)
    hour = np.clip(rng.normal(13, 3.5, m), 0, 23.99)
    ts = start + pd.to_timedelta(day, unit="D") + pd.to_timedelta(hour * 3600, unit="s")
    amount = np.round(base[owner] * rng.lognormal(0, 0.5, m), 2)
    kind = rng.choice(4, m, p=[0.55, 0.15, 0.15, 0.15])  # out, cash in, cash out, ext in

    internal = rng.random(m) < 0.7
    other = rng.integers(0, n - 1, m)
    other = other + (other >= owner)
    ext_idx = rng.integers(0, cfg.n_external, m)
    counterparty = np.where(internal, customers[other], externals[ext_idx])
    cp_country = np.where(internal, home, ext_country[ext_idx])
    me = customers[owner]

    sender = np.select([kind == 0, kind == 1, kind == 2, kind == 3],
                       [me, "CASH", me, externals[ext_idx]], default="")
    receiver = np.select([kind == 0, kind == 1, kind == 2, kind == 3],
                         [counterparty, me, "CASH", me], default="")
    channel = np.where((kind == 1) | (kind == 2), "cash", "transfer")
    s_country = np.where(kind == 3, ext_country[ext_idx], home)
    r_country = np.where(kind == 0, cp_country, home)

    normal = pd.DataFrame({
        "timestamp": ts,
        "sender": sender,
        "receiver": receiver,
        "amount": amount,
        "channel": channel,
        "sender_country": s_country,
        "receiver_country": r_country,
        "typology": "none",
    })

    # ---- injected typologies ------------------------------------------
    b = _Builder()
    candidates = np.flatnonzero(~is_business)
    k = max(len(TYPOLOGIES), round(n * cfg.typology_share))
    chosen = rng.choice(candidates, size=min(k, candidates.size), replace=False)
    ext_counter = [0]

    def new_external(country: str = home) -> str:
        ext_counter[0] += 1
        return f"Y{ext_counter[0]:05d}"

    def rand_time(day_lo: float, day_hi: float) -> pd.Timestamp:
        d = rng.uniform(day_lo, day_hi)
        return start + pd.Timedelta(days=d)

    thr = cfg.reporting_threshold
    for j, idx in enumerate(chosen):
        typ = TYPOLOGIES[j % len(TYPOLOGIES)]
        cid = customers[idx]
        anchor = rng.uniform(5, cfg.days - 20)

        if typ == "structuring":
            for _ in range(rng.integers(5, 11)):
                b.add(rand_time(anchor, anchor + rng.uniform(3, 7)), "CASH", cid,
                      thr * rng.uniform(0.85, 0.99), "cash", home, home, typ)

        elif typ == "pass_through":
            for _ in range(rng.integers(3, 6)):
                t_in = rand_time(anchor, anchor + 15)
                amt = rng.uniform(5_000, 50_000)
                origin = new_external()
                dst = new_external()
                dst_country = str(rng.choice(_FOREIGN))
                b.add(t_in, origin, cid, amt, "transfer", home, home, typ)
                b.add(t_in + pd.Timedelta(hours=rng.uniform(1, 20)), cid, dst,
                      amt * rng.uniform(0.9, 0.99), "transfer", home, dst_country, typ)

        elif typ == "fan_in":
            total = 0.0
            for _ in range(rng.integers(15, 31)):
                amt = rng.uniform(200, 2_000)
                total += amt
                b.add(rand_time(anchor, anchor + 7), new_external(), cid, amt,
                      "transfer", home, home, typ)
            b.add(start + pd.Timedelta(days=anchor + 8), cid, "CASH",
                  total * rng.uniform(0.8, 0.95), "cash", home, home, typ)

        elif typ == "round_trip":
            partners = rng.choice(np.setdiff1d(np.arange(n), [idx]), size=rng.integers(2, 4),
                                  replace=False)
            chain = [cid, *customers[partners], cid]
            for _ in range(rng.integers(2, 4)):
                t = rand_time(anchor, anchor + 10)
                amt = rng.uniform(5_000, 40_000)
                for s, r in pairwise(chain):
                    b.add(t, s, r, amt, "transfer", home, home, typ)
                    t = t + pd.Timedelta(hours=rng.uniform(2, 48))
                    amt *= rng.uniform(0.95, 0.99)

        elif typ == "high_risk_corridor":
            for _ in range(rng.integers(4, 9)):
                country = str(rng.choice(cfg.high_risk_countries))
                b.add(rand_time(anchor, anchor + 20), cid, new_external(country),
                      rng.uniform(3_000, 20_000), "transfer", home, country, typ)

    injected = pd.DataFrame(b.cols)
    df = pd.concat([normal, injected], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.round("s")
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    df["currency"] = cfg.currency
    df["txn_id"] = [f"T{i:07d}" for i in range(len(df))]
    return df[COLUMNS]
