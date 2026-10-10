import pandas as pd

from amlwatch import TYPOLOGIES, Monitor, RuleConfig, generate_transactions
from amlwatch.cli import main


def test_generator_is_deterministic_and_labelled():
    a = generate_transactions(n_customers=200, days=30, seed=1)
    b = generate_transactions(n_customers=200, days=30, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["typology"]) - {"none"} == set(TYPOLOGIES)
    assert a["timestamp"].is_monotonic_increasing
    assert a["txn_id"].is_unique


def test_end_to_end_detection_quality():
    result = Monitor().run(generate_transactions(n_customers=600, seed=3))
    ev = result.evaluate()
    assert ev.loc["all alerts", "recall"] >= 0.9
    assert ev.loc["all alerts", "precision"] >= 0.6
    assert result.customers["risk_score"].between(0, 100).all()
    assert set(result.customers["status"]) <= {"alert", "anomaly_review", "clear"}


def test_config_from_toml(tmp_path):
    p = tmp_path / "rules.toml"
    p.write_text('structuring_min_count = 5\nhigh_risk_countries = ["AA"]\n[weights]\nfan_in = 0.9\n')
    cfg = RuleConfig.from_toml(p)
    assert cfg.structuring_min_count == 5
    assert cfg.high_risk_countries == ("AA",)
    assert cfg.weights["fan_in"] == 0.9 and cfg.weights["structuring"] == 0.6


def test_cli_generate_and_scan(tmp_path, capsys):
    data = tmp_path / "t.csv"
    report = tmp_path / "r.html"
    alerts = tmp_path / "a.csv"
    assert main(["generate", "--out", str(data), "--customers", "200", "--days", "45"]) == 0
    assert main(["scan", str(data), "--report", str(report), "--alerts", str(alerts)]) == 0
    out = capsys.readouterr().out
    assert "Alerted customers" in out and "precision" in out
    assert "<table>" in report.read_text()
    assert len(pd.read_csv(alerts)) > 0
