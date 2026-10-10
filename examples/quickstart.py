"""Generate a synthetic ledger, run monitoring and write a report."""

import pandas as pd

from amlwatch import Monitor, RuleConfig, generate_transactions

pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 90)

transactions = generate_transactions(n_customers=1_000, days=90, seed=42)
print(f"{len(transactions):,} transactions generated")

result = Monitor(RuleConfig.from_toml("examples/rules.toml")).run(transactions)

print("\nTop of the alert queue")
print(result.alerted[["risk_score", "rules", "anomaly"]].head(10))

print("\nEvidence for the highest-risk customer")
top = result.alerted.index[0]
for detail in result.alerts.loc[result.alerts["customer"] == top, "detail"]:
    print(f"  - {detail}")

print("\nDetection quality against injected labels")
print(result.evaluate().round(2))

result.to_html("report.html")
print("\nReport written to report.html")
