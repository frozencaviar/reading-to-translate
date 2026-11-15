#!/usr/bin/env python3
"""Export mixed-effects attention-shrinkage table for the decomposition plot."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import run_mixed_effects_yiyu_analysis as mix


ATT_FEATURES = {
    "src": ["x2x", "x2cont", "x2eos", "x_ent", "cont2x", "tgtseq2x"],
    "tgt": ["eos_xatt", "y2srcseq_ent", "y2y", "y2cont", "y_ent"],
}

ROW_ORDER = [
    ("src", "log_TrtS", "token", "Source reading (token)"),
    ("src", "log_TrtS", "segment", "Source reading (segment)"),
    ("tgt", "log_TrtT", "token", "Target reading (token)"),
    ("tgt", "log_TrtT", "segment", "Target reading (segment)"),
    ("tgt", "log_Dur", "token", "Target duration (token)"),
    ("tgt", "log_Dur", "segment", "Target duration (segment)"),
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=script_dir / "attention-norm-yiyu-positive-dur")
    parser.add_argument("--mixed-dir", type=Path, default=script_dir / "figures_yiyu_positive_dur_mixed")
    parser.add_argument("--item-col", default="sentence_item_id")
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--min-rows", type=int, default=40)
    return parser.parse_args()


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else np.nan


def values_from_fig3(fig3_all: pd.DataFrame, side: str, label: str, level: str, delta: str) -> list[float]:
    sub = fig3_all[
        (fig3_all["side"].eq(side))
        & (fig3_all["label"].eq(label))
        & (fig3_all["level"].eq(level))
        & (fig3_all["delta"].eq(delta))
        & (fig3_all["feature"].isin(ATT_FEATURES[side]))
    ].copy()
    sub["metric"] = pd.to_numeric(sub["metric"], errors="coerce")
    return [float(v) for v in sub["metric"].dropna().tolist()]


def fit_target_lm_attention(
    data: dict[str, dict[str, pd.DataFrame]],
    item_col: str,
    maxiter: int,
    min_rows: int,
) -> tuple[dict[tuple[str, str], list[float]], pd.DataFrame]:
    """Fit target attention features after target LM surprisal.

    The grouped residual-attention mixed table controls target-side attention by
    NMT cross entropy (`xent`). This helper adds the target LM-surprisal
    counterpart needed for the LM panel in the shrinkage decomposition.
    """
    status_rows: list[dict[str, object]] = []
    target_values: dict[tuple[str, str], list[float]] = {}
    for level in mix.LEVELS:
        df_level = mix.filter_group(data["tgt"][level], "all")
        for label in ["log_TrtT", "log_Dur"]:
            metrics: list[float] = []
            for predictor in ATT_FEATURES["tgt"]:
                rec, problems = mix.fit_pair(
                    df_level,
                    label,
                    mix.CONTROL_FEATURES["tgt"][level] + ["tgt_surp"],
                    predictor,
                    item_col,
                    maxiter,
                    min_rows,
                )
                for problem in problems:
                    status_rows.append(
                        {
                            **problem,
                            "group": "all",
                            "side": "tgt",
                            "level": level,
                            "label": label,
                            "delta": "*ctrl+tgt_surp",
                        }
                    )
                if rec is not None:
                    metrics.append(float(rec["metric"]))
                    status_rows.append(
                        {
                            "predictor": rec["predictor"],
                            "status": "ok",
                            "group": "all",
                            "side": "tgt",
                            "level": level,
                            "label": label,
                            "delta": "*ctrl+tgt_surp",
                            "n_rows": rec["n_rows"],
                            "n_participants": rec["n_participants"],
                            "n_items": rec["n_items"],
                            "base_converged": rec["base_converged"],
                            "predictor_converged": rec["predictor_converged"],
                            "base_model_kind": rec["base_model_kind"],
                            "predictor_model_kind": rec["predictor_model_kind"],
                        }
                    )
            target_values[(label, level)] = metrics
    return target_values, pd.DataFrame(status_rows)


def build_shrinkage_table(
    fig3_all: pd.DataFrame,
    target_lm_values: dict[tuple[str, str], list[float]],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for side, label, level, cell in ROW_ORDER:
        control_values = values_from_fig3(fig3_all, side, label, level, "*ctrl")
        control_mean = mean_or_nan(control_values)

        if side == "src":
            lm_values = values_from_fig3(fig3_all, side, label, level, "*ctrl+surp")
        else:
            lm_values = target_lm_values.get((label, level), [])
        lm_mean = mean_or_nan(lm_values)
        records.append(
            {
                "baseline_model": "LM",
                "side": side,
                "target": label,
                "level": level,
                "cell": cell,
                "control_only": control_mean,
                "control_plus_surprisal": lm_mean,
                "difference": control_mean - lm_mean if np.isfinite(control_mean) and np.isfinite(lm_mean) else np.nan,
                "control_values_json": json.dumps(control_values, ensure_ascii=False),
                "surprisal_values_json": json.dumps(lm_values, ensure_ascii=False),
            }
        )

        if side == "tgt":
            nmt_values = values_from_fig3(fig3_all, side, label, level, "*ctrl+surp")
            nmt_mean = mean_or_nan(nmt_values)
            records.append(
                {
                    "baseline_model": "NMT",
                    "side": side,
                    "target": label,
                    "level": level,
                    "cell": cell,
                    "control_only": control_mean,
                    "control_plus_surprisal": nmt_mean,
                    "difference": control_mean - nmt_mean if np.isfinite(control_mean) and np.isfinite(nmt_mean) else np.nan,
                    "control_values_json": json.dumps(control_values, ensure_ascii=False),
                    "surprisal_values_json": json.dumps(nmt_values, ensure_ascii=False),
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore")
    fig3 = pd.read_csv(args.mixed_dir / "figure3_delta_llh_att_grouped.tsv", sep="\t")
    fig3_all = fig3[fig3["group"].eq("all")].copy()
    data = mix.load_data(args.data_dir)
    target_lm_values, status = fit_target_lm_attention(data, args.item_col, args.maxiter, args.min_rows)
    shrinkage = build_shrinkage_table(fig3_all, target_lm_values)

    args.mixed_dir.mkdir(parents=True, exist_ok=True)
    shrinkage_path = args.mixed_dir / "figure4_att_shrinkage.tsv"
    status_path = args.mixed_dir / "table_mixed_att_shrinkage_extra_fit_status.tsv"
    shrinkage.to_csv(shrinkage_path, sep="\t", index=False)
    status.to_csv(status_path, sep="\t", index=False)
    print(shrinkage_path)
    print(status_path)


if __name__ == "__main__":
    main()
