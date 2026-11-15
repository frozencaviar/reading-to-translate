#!/usr/bin/env python3
"""Export AIC-gain tables from mixed-effects likelihood comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "figures_yiyu_positive_dur_mixed" / "table_mixed_effects_all_tests.tsv",
    )
    parser.add_argument("--outdir", type=Path, default=script_dir / "figures_yiyu_positive_dur_mixed")
    return parser.parse_args()


def add_aic_columns(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    delta_ll = pd.to_numeric(out["llf_predictor"], errors="coerce") - pd.to_numeric(out["llf_control"], errors="coerce")
    n_rows = pd.to_numeric(out["n_rows"], errors="coerce").replace(0, np.nan)

    # The augmented model adds one fixed-effect parameter while keeping the
    # random-effects structure unchanged, so Delta AIC = 2 - 2*Delta LL.
    out["delta_aic"] = 2.0 - 2.0 * delta_ll
    out["aic_gain"] = -out["delta_aic"]
    out["aic_gain_per_obs"] = out["aic_gain"] / n_rows
    out["metric"] = out["aic_gain_per_obs"]
    return out


def main() -> None:
    args = parse_args()
    table = pd.read_csv(args.input, sep="\t")
    required = {"llf_control", "llf_predictor", "n_rows", "delta"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = add_aic_columns(table)
    args.outdir.mkdir(parents=True, exist_ok=True)
    all_path = args.outdir / "table_mixed_aic_all_tests.tsv"
    fig5_path = args.outdir / "figure5_aic_gain_grouped.tsv"
    fig3_path = args.outdir / "figure3_aic_gain_att_grouped.tsv"

    out.to_csv(all_path, sep="\t", index=False)
    out[out["delta"].eq("*ctrl")].to_csv(fig5_path, sep="\t", index=False)
    out[out["delta"].isin(["*ctrl", "*ctrl+surp"])].to_csv(fig3_path, sep="\t", index=False)
    print(all_path)
    print(fig5_path)
    print(fig3_path)


if __name__ == "__main__":
    main()
