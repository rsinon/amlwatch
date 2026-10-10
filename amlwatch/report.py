"""Self-contained HTML report for investigators."""

from __future__ import annotations

import html
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .monitor import MonitoringResult

_CSS = """
body{font-family:Arial,Helvetica,sans-serif;color:#111;margin:2rem auto;max-width:1100px;padding:0 1rem;line-height:1.45}
h1{font-size:1.6rem;margin-bottom:.2rem}h2{font-size:1.15rem;margin-top:2rem;border-bottom:1px solid #999;padding-bottom:.2rem}
.meta{color:#555;font-size:.9rem}
.kpis{display:flex;gap:1rem;flex-wrap:wrap;margin:1.2rem 0}
.kpi{border:1px solid #ccc;padding:.7rem 1rem;min-width:150px}
.kpi b{display:block;font-size:1.5rem}
table{border-collapse:collapse;width:100%;font-size:.88rem;margin-top:.5rem}
thead th{border-top:2px solid #111;border-bottom:1px solid #111;background:#eee;text-align:left;padding:.35rem .5rem}
tbody td{padding:.3rem .5rem;vertical-align:top}
tbody tr:last-child td{border-bottom:2px solid #111}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.note{font-size:.8rem;color:#555;font-style:italic;margin-top:.3rem}
"""


def _table(df: pd.DataFrame, num_cols: set[str] = frozenset()) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            cls = ' class="num"' if c in num_cols else ""
            cells.append(f"<td{cls}>{html.escape('' if pd.isna(v) else str(v))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render_html(result: MonitoringResult, top: int = 50) -> str:
    s = result.summary()
    tx = result.transactions
    period = f"{tx['timestamp'].min():%d %b %Y} to {tx['timestamp'].max():%d %b %Y}"
    generated = datetime.now(UTC).strftime("%d %b %Y %H:%M UTC")

    kpis = "".join(
        f'<div class="kpi">{label}<b>{value:,}</b></div>'
        for label, value in [
            ("Transactions", s["transactions"]),
            ("Customers", s["customers"]),
            ("Customers alerted", s["alerted_customers"]),
            ("Anomaly review", s["anomaly_review"]),
        ]
    )

    rule_counts = pd.DataFrame(
        sorted(s["alerts_by_rule"].items(), key=lambda kv: -kv[1]), columns=["Rule", "Customers"])

    queue = result.alerted.head(top).reset_index()
    if len(result.alerts):
        detail = result.alerts.groupby("customer")["detail"].apply(lambda d: "; ".join(d))
        queue["Evidence"] = queue["customer"].map(detail)
    else:
        queue["Evidence"] = ""
    queue = queue.assign(
        risk=queue["risk_score"].map("{:.1f}".format),
        anomaly=queue["anomaly"].map("{:.2f}".format),
    )[["customer", "risk", "rules", "anomaly", "Evidence"]]
    queue.columns = ["Customer", "Risk score", "Rules", "Anomaly pct", "Evidence"]

    review = result.customers[result.customers["status"] == "anomaly_review"].head(20).reset_index()
    review = review.assign(
        anomaly=review["anomaly"].map("{:.3f}".format),
        value_in=review["value_in"].map("{:,.0f}".format),
        value_out=review["value_out"].map("{:,.0f}".format),
    )[["customer", "anomaly", "n_in", "n_out", "value_in", "value_out"]]
    review.columns = ["Customer", "Anomaly pct", "Txns in", "Txns out", "Value in", "Value out"]

    sections = [
        "<h1>Transaction monitoring report</h1>",
        f'<p class="meta">Period {period}. Generated {generated} by amlwatch.</p>',
        f'<div class="kpis">{kpis}</div>',
        "<h2>1. Alerts by rule</h2>",
        "<p>Table 1. Customers flagged by each rule</p>",
        _table(rule_counts, {"Customers"}),
        "<h2>2. Alert queue</h2>",
        f"<p>Table 2. Alerted customers ranked by risk score (top {top})</p>",
        _table(queue, {"Risk score", "Anomaly pct"}),
        (
            '<p class="note">Risk score combines weighted rule evidence with the customer\'s '
            "anomaly percentile. It orders the queue; it is not a finding of wrongdoing.</p>"
        ),
        "<h2>3. Anomaly review</h2>",
        "<p>Table 3. Customers with no rule alert but unusually atypical behaviour</p>",
        _table(review, {"Anomaly pct", "Txns in", "Txns out", "Value in", "Value out"}),
    ]

    evaluation = result.evaluate()
    if evaluation is not None:
        ev = evaluation.reset_index()
        for col in ("precision", "recall", "f1"):
            ev[col] = ev[col].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
        ev["positives"] = ev["positives"].map(lambda v: "" if pd.isna(v) else f"{int(v)}")
        sections += [
            "<h2>4. Detection quality</h2>",
            "<p>Table 4. Alerts measured against labelled typologies</p>",
            _table(ev, {"flagged", "positives", "true_positives", "precision", "recall", "f1"}),
            (
                '<p class="note">Only available when the input carries a typology label, '
                "for example synthetic data from amlwatch.generate_transactions.</p>"
            ),
        ]

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Monitoring report</title><style>{_CSS}</style></head><body>"
        + "\n".join(sections) + "</body></html>"
    )
