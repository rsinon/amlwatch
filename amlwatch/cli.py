"""Command-line interface: ``amlwatch generate | scan``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .config import RuleConfig
from .monitor import Monitor
from .synth import SynthConfig, generate_transactions


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def cmd_generate(args: argparse.Namespace) -> int:
    df = generate_transactions(SynthConfig(
        n_customers=args.customers, days=args.days, seed=args.seed,
        typology_share=args.typology_share))
    df.to_csv(args.out, index=False)
    labelled = (df["typology"] != "none").sum()
    print(f"Wrote {len(df):,} transactions ({labelled:,} from injected typologies) to {args.out}")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    cfg = RuleConfig.from_toml(args.config) if args.config else RuleConfig()
    result = Monitor(cfg).run(_read(args.input))
    s = result.summary()
    print(f"Scanned {s['transactions']:,} transactions across {s['customers']:,} customers")
    print(f"Alerted customers: {s['alerted_customers']}  |  anomaly review: {s['anomaly_review']}")
    for rule, n in sorted(s["alerts_by_rule"].items(), key=lambda kv: -kv[1]):
        print(f"  {rule:<20} {n:>5}")

    if args.alerts:
        out = result.alerts.drop(columns=["txn_ids"]).assign(
            txn_ids=result.alerts["txn_ids"].str.join(" "))
        out.to_csv(args.alerts, index=False)
        print(f"Alerts written to {args.alerts}")
    if args.report:
        result.to_html(args.report)
        print(f"Report written to {args.report}")

    ev = result.evaluate()
    if ev is not None:
        row = ev.loc["all alerts"]
        print(f"Against labels: precision {row['precision']:.2f}, recall {row['recall']:.2f}, "
              f"F1 {row['f1']:.2f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="amlwatch", description="Open-source AML transaction monitoring")
    p.add_argument("--version", action="version", version=f"amlwatch {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="create a labelled synthetic dataset")
    g.add_argument("--out", type=Path, default=Path("transactions.csv"))
    g.add_argument("--customers", type=int, default=1_000)
    g.add_argument("--days", type=int, default=90)
    g.add_argument("--typology-share", type=float, default=0.03)
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("scan", help="run monitoring over a CSV or Parquet file")
    s.add_argument("input", type=Path)
    s.add_argument("--config", type=Path, help="TOML file of rule thresholds")
    s.add_argument("--report", type=Path, help="write an HTML report here")
    s.add_argument("--alerts", type=Path, help="write the alert table as CSV here")
    s.set_defaults(func=cmd_scan)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
