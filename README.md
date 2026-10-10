# amlwatch

[![CI](https://github.com/rsinon/amlwatch/actions/workflows/ci.yml/badge.svg)](https://github.com/rsinon/amlwatch/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Licence](https://img.shields.io/badge/licence-MIT-green)

Open-source anti-money-laundering (AML) transaction monitoring in Python.

`amlwatch` scans a table of transactions, raises explainable alerts for known laundering typologies, ranks customers by risk and produces an investigator-ready HTML report. It ships with a synthetic data generator that injects labelled typologies, so you can measure detection quality without touching real customer data.

Most transaction monitoring software is closed, expensive and hard to inspect. This project is meant for the people who need to understand how detection works: compliance analysts, fintech engineers, students, researchers and smaller institutions evaluating their options.

## Quick start

```bash
pip install git+https://github.com/rsinon/amlwatch.git

amlwatch generate --out transactions.csv          # 1,000 synthetic customers, 90 days
amlwatch scan transactions.csv --report report.html --alerts alerts.csv
```

```text
Scanned 60,116 transactions across 1,000 customers
Alerted customers: 53  |  anomaly review: 2
  round_trip              23
  pass_through            14
  fan_in                   9
  high_risk_corridor       8
  structuring              7
Report written to report.html
Against labels: precision 0.89, recall 1.00, F1 0.94
```

Or from Python:

```python
from amlwatch import Monitor, generate_transactions

transactions = generate_transactions(n_customers=1_000, seed=42)
result = Monitor().run(transactions)

result.alerted.head()      # customers ranked by risk score
result.alerts              # one row per alert, with the transaction IDs behind it
result.evaluate()          # precision and recall against the injected labels
result.to_html("report.html")
```

## What it detects

Each rule is a short, readable function in [`rules.py`](amlwatch/rules.py). Every alert records the transactions that triggered it and a plain-language explanation, for example *"3 of 7 inflows of 3,000+ (76% of that value) forwarded at 90%+, median 6.2 hours later"*.

| Rule | Typology | Logic |
|---|---|---|
| `structuring` | Cash deposits split to stay under a reporting threshold | At least 3 cash deposits between 80% and 100% of the threshold within 7 days |
| `pass_through` | Account used to move funds on quickly (layering, mule accounts) | Large inflows forwarded within 48 hours at 90%+ of value, covering most of the account's large inflow value |
| `fan_in` | Many people paying one account (collection mules, scam proceeds) | 12+ distinct senders of small amounts within 7 days |
| `high_risk_corridor` | Flows to or from listed jurisdictions | 2+ transfers over a minimum amount involving a configured country list |
| `round_trip` | Value that leaves and returns through a chain of accounts | Time-ordered cycles A → B → … → A within 14 days that keep 85%+ of value at each hop |

On top of the rules, an isolation forest scores how unusual each customer's overall behaviour is (volumes, cash share, counterparty spread, foreign share, flow-through ratio, night-time activity). The anomaly percentile does two jobs:

1. it contributes to the **risk score** (0 to 100) that orders the alert queue, and
2. it places customers with no rule alert but extreme behaviour in a separate **anomaly review** list, which catches patterns no rule was written for.

Rule evidence is combined with a noisy-OR, so a customer hit by two independent rules ranks above one hit by a single rule, without the score exceeding its ceiling.

## Measuring detection quality

The generator creates ordinary activity for individuals and businesses, then injects the five typologies into about 3% of customers and labels every injected transaction. `result.evaluate()` reports:

- overall precision and recall at customer level,
- recall within each typology, and
- precision of each rule.

Results on the default synthetic scenario (1,000 customers, 90 days, seed 42):

| Scope | Flagged | True positives | Precision | Recall |
|---|---:|---:|---:|---:|
| All alerts | 53 | 47 | 0.89 | 1.00 |
| Rule: round_trip | 23 | 23 | 1.00 | |
| Rule: pass_through | 14 | 14 | 1.00 | |
| Rule: fan_in | 9 | 6 | 0.67 | |
| Rule: high_risk_corridor | 8 | 6 | 0.75 | |
| Rule: structuring | 7 | 6 | 0.86 | |

**Read these numbers with care.** They show that the rules detect the typologies as designed and that false positives arise from realistic sources (busy business accounts, legitimate counterparties in listed countries). They are not an estimate of performance on real data. Real laundering is adaptive, labels are scarce, and production systems typically run with far lower precision. The value of the harness is in comparing rule changes against a fixed baseline before they reach a live book.

## Using your own data

`amlwatch` needs one row per transaction:

| Column | Required | Notes |
|---|---|---|
| `txn_id` | yes | unique identifier |
| `timestamp` | yes | any format pandas can parse |
| `sender`, `receiver` | yes | account identifiers; use one fixed ID (default `CASH`) for cash |
| `amount` | yes | non-negative; direction comes from sender and receiver |
| `channel` | yes | `transfer` or `cash` |
| `sender_country`, `receiver_country` | no | ISO codes, used by the corridor rule and features |
| `sender_is_customer`, `receiver_is_customer` | no | booleans marking your institution's own accounts |

If the customer flags are absent, accounts matching `customer_pattern` (default `^C\d+$`) are treated as customers. CSV and Parquet both work from the command line.

## Configuration

Every threshold lives in [`RuleConfig`](amlwatch/config.py) and can be set from a TOML file:

```bash
amlwatch scan transactions.csv --config examples/rules.toml
```

```toml
reporting_threshold = 10000
structuring_min_count = 3
high_risk_countries = ["ZZ", "ZY"]   # placeholders: supply your own list

[weights]
round_trip = 0.7
structuring = 0.6
```

The defaults are illustrative. Calibrate thresholds to your own customer segments, products and regulatory reporting limits, and document the rationale, as supervisors will expect.

## Project layout

```text
amlwatch/
  synth.py      synthetic ledger with labelled typologies
  schema.py     input validation and normalisation
  rules.py      the five detection rules
  scoring.py    customer features and isolation-forest anomaly score
  monitor.py    pipeline, risk score and triage status
  evaluate.py   precision and recall against labels
  report.py     self-contained HTML report
  cli.py        command-line interface
tests/          unit tests per rule and an end-to-end test
```

## Development

```bash
git clone https://github.com/rsinon/amlwatch.git
cd amlwatch
pip install -e ".[dev]"
pytest
ruff check amlwatch tests
```

## Roadmap

- Customer segmentation so thresholds can differ for individuals and businesses
- Rapid movement of funds across many accounts (graph-based layering beyond simple cycles)
- Alert suppression and case management fields for repeat alerts
- Benchmark against the public AMLworld synthetic datasets
- A small dashboard for alert review

Issues and pull requests are welcome, particularly new typologies with a short description of the behaviour they target.

## Disclaimer

`amlwatch` is an educational and research tool. It is not a certified compliance system, and its output is not a determination that any person has committed an offence. Institutions remain responsible for their own monitoring obligations under applicable law.

## Author

Built by **Rudy Sinon**, Senior Statistician and principal of Atlas Intelligence, a data intelligence consultancy.

Released under the [MIT licence](LICENSE).
