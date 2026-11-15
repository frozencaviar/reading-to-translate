#!/usr/bin/env python3
"""Compute VIF diagnostics for mixed-model fixed effects."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import run_mixed_effects_yiyu_analysis as mix


FEATURES = mix.FEATURES
ROWS = [
    ("length", "Length"),
    ("frequency", "Frequency"),
    ("position", "Position"),
    ("direction", "Direction"),
    ("predictor", "Predictor"),
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=script_dir / "attention-norm-yiyu-positive-dur")
    parser.add_argument("--outdir", type=Path, default=script_dir / "figures_yiyu_positive_dur_mixed")
    return parser.parse_args()


def vif_for_matrix(matrix: pd.DataFrame) -> dict[str, float]:
    x = matrix.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    x = x.loc[:, x.nunique(dropna=True) > 1]
    values: dict[str, float] = {}
    for col in x.columns:
        y = x[col].to_numpy(dtype=float)
        others = x.drop(columns=[col]).to_numpy(dtype=float)
        if others.shape[1] == 0 or np.nanvar(y) <= 0:
            values[col] = np.nan
            continue
        design = np.column_stack([np.ones(len(others)), others])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ coef
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot <= 0:
            values[col] = np.nan
            continue
        r2 = min(max(1.0 - ss_res / ss_tot, 0.0), 0.999999)
        values[col] = float(1.0 / (1.0 - r2))
    return values


def design_for_feature(df: pd.DataFrame, side: str, feature: str) -> tuple[pd.DataFrame, dict[str, str | list[str]]]:
    controls = {
        "length": f"{side}_seg_len",
        "frequency": f"{side}_freq",
        "position": f"{side}_posq",
        "predictor": feature,
    }
    required = list(controls.values()) + ["lang_pair"]
    work = df.dropna(subset=required).copy()
    direction = pd.get_dummies(work["lang_pair"].astype(str), prefix="direction", drop_first=True, dtype=float)
    design = work[list(controls.values())].copy()
    design = pd.concat([design, direction], axis=1)
    aliases: dict[str, str | list[str]] = {
        "length": controls["length"],
        "frequency": controls["frequency"],
        "position": controls["position"],
        "predictor": controls["predictor"],
        "direction": list(direction.columns),
    }
    return design, aliases


def main() -> None:
    args = parse_args()
    data = mix.load_data(args.data_dir)
    records: list[dict[str, object]] = []
    for side in mix.SIDES:
        for level in mix.LEVELS:
            df = mix.filter_group(data[side][level], "all")
            for feature in FEATURES[side]:
                design, aliases = design_for_feature(df, side, feature)
                vif_values = vif_for_matrix(design)
                for row_key, _ in ROWS:
                    alias = aliases[row_key]
                    if isinstance(alias, list):
                        vals = [vif_values.get(name, np.nan) for name in alias]
                        vif = float(np.nanmax(vals)) if any(np.isfinite(vals)) else np.nan
                        vif_term = ",".join(alias)
                    else:
                        vif = vif_values.get(alias, np.nan)
                        vif_term = alias
                    records.append(
                        {
                            "side": side,
                            "level": level,
                            "feature": feature,
                            "vif_row": row_key,
                            "vif_label": dict(ROWS)[row_key],
                            "vif_term": vif_term,
                            "vif_value": vif,
                            "n_rows": len(design.dropna()),
                        }
                    )

    out = pd.DataFrame(records)
    args.outdir.mkdir(parents=True, exist_ok=True)
    path = args.outdir / "figure4_vif.tsv"
    out.to_csv(path, sep="\t", index=False)
    print(path)


if __name__ == "__main__":
    main()
