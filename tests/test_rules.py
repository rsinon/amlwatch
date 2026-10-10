import pandas as pd
import pytest

from amlwatch import Monitor, RuleConfig, SchemaError
from amlwatch.rules import fan_in, high_risk_corridor, pass_through, round_trip, structuring
from amlwatch.schema import prepare

CFG = RuleConfig()


def frame(rows):
    cols = ["timestamp", "sender", "receiver", "amount", "channel", "sender_country", "receiver_country"]
    df = pd.DataFrame(rows, columns=cols)
    df["txn_id"] = [f"T{i}" for i in range(len(df))]
    return prepare(df, CFG)


def t(day, hour=12):
    return pd.Timestamp("2026-03-01") + pd.Timedelta(days=day, hours=hour)


def test_structuring_flags_repeated_sub_threshold_deposits():
    df = frame([(t(d), "CASH", "C00001", 9_500, "cash", "SC", "SC") for d in range(4)])
    alerts = structuring(df, CFG)
    assert [a.customer for a in alerts] == ["C00001"]
    assert len(alerts[0].txn_ids) == 4


def test_structuring_ignores_spread_out_deposits():
    df = frame([(t(d * 10), "CASH", "C00001", 9_500, "cash", "SC", "SC") for d in range(4)])
    assert structuring(df, CFG) == []


def test_structuring_ignores_amounts_over_threshold():
    df = frame([(t(d), "CASH", "C00001", 12_000, "cash", "SC", "SC") for d in range(5)])
    assert structuring(df, CFG) == []


def test_pass_through_flags_quick_forwarding():
    rows = []
    for d in range(3):
        rows.append((t(d * 3, 9), "X1", "C00001", 20_000, "transfer", "SC", "SC"))
        rows.append((t(d * 3, 15), "C00001", "X2", 19_000, "transfer", "SC", "GB"))
    alerts = pass_through(frame(rows), CFG)
    assert [a.customer for a in alerts] == ["C00001"]


def test_pass_through_ignores_slow_or_partial_forwarding():
    rows = []
    for d in range(3):
        rows.append((t(d * 5, 9), "X1", "C00001", 20_000, "transfer", "SC", "SC"))
        rows.append((t(d * 5 + 4, 9), "C00001", "X2", 19_000, "transfer", "SC", "GB"))  # 4 days later
        rows.append((t(d * 5, 10), "C00001", "X3", 5_000, "transfer", "SC", "SC"))      # 25% only
    assert pass_through(frame(rows), CFG) == []


def test_fan_in_counts_distinct_senders():
    rows = [(t(i % 5), f"X{i}", "C00001", 500, "transfer", "SC", "SC") for i in range(15)]
    alerts = fan_in(frame(rows), CFG)
    assert alerts and alerts[0].customer == "C00001"
    assert "15 distinct senders" in alerts[0].detail


def test_fan_in_repeat_sender_does_not_count_twice():
    rows = [(t(i % 5), "X1", "C00001", 500, "transfer", "SC", "SC") for i in range(30)]
    assert fan_in(frame(rows), CFG) == []


def test_high_risk_corridor():
    rows = [(t(i), "C00001", f"X{i}", 5_000, "transfer", "SC", "ZZ") for i in range(3)]
    alerts = high_risk_corridor(frame(rows), CFG)
    assert alerts[0].customer == "C00001" and "ZZ" in alerts[0].detail


def test_high_risk_corridor_respects_minimum_count():
    rows = [(t(0), "C00001", "X1", 5_000, "transfer", "SC", "ZZ")]
    assert high_risk_corridor(frame(rows), CFG) == []


def test_round_trip_detects_time_ordered_cycle():
    rows = [
        (t(0, 9), "C00001", "C00002", 20_000, "transfer", "SC", "SC"),
        (t(0, 15), "C00002", "C00003", 19_500, "transfer", "SC", "SC"),
        (t(1, 9), "C00003", "C00001", 19_000, "transfer", "SC", "SC"),
    ]
    flagged = {a.customer for a in round_trip(frame(rows), CFG)}
    assert flagged == {"C00001", "C00002", "C00003"}


def test_round_trip_requires_forward_time_order():
    rows = [
        (t(2), "C00001", "C00002", 20_000, "transfer", "SC", "SC"),
        (t(1), "C00002", "C00003", 19_500, "transfer", "SC", "SC"),
        (t(0), "C00003", "C00001", 19_000, "transfer", "SC", "SC"),
    ]
    assert round_trip(frame(rows), CFG) == []


def test_schema_rejects_missing_columns():
    with pytest.raises(SchemaError):
        Monitor().run(pd.DataFrame({"amount": [1.0]}))


def test_explicit_customer_flags_override_pattern():
    df = pd.DataFrame({
        "txn_id": ["a", "b", "c", "d"],
        "timestamp": [t(0), t(1), t(2), t(3)],
        "sender": ["CASH"] * 4,
        "receiver": ["acct-77"] * 4,
        "amount": [9_500] * 4,
        "channel": ["CASH"] * 4,
        "sender_is_customer": [False] * 4,
        "receiver_is_customer": [True] * 4,
    })
    result = Monitor(rules=["structuring"]).run(df)
    assert list(result.alerts["customer"]) == ["acct-77"]
