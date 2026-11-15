#!/usr/bin/env python3
"""Export fixed-effect coefficient tables for mixed-effects heatmaps."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

import run_mixed_effects_yiyu_analysis as mixed

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=script_dir / "attention-norm-yiyu-positive-dur")
    parser.add_argument("--outdir", type=Path, default=script_dir / "figures_yiyu_positive_dur_mixed")
    parser.add_argument("--item-col", default="sentence_item_id")
    parser.add_argument("--maxiter", type=int, default=300)
    return parser.parse_args()


def fit_result(df: pd.DataFrame, label: str, features: list[str], item_col: str, maxiter: int):
    formula = mixed.build_formula(label, features)
    work = df.dropna(subset=list(dict.fromkeys(features + [label, "lang_pair", "translator_id", item_col]))).copy()
    work["_all_group"] = "all"
    model = smf.mixedlm(
        formula,
        data=work,
        groups=work["_all_group"],
        re_formula="0",
        vc_formula={
            "participant": "0 + C(translator_id)",
            "item": f"0 + C({item_col})",
        },
    )
    best = None
    for method in ("lbfgs", "bfgs", "cg"):
        result = model.fit(reml=False, method=method, maxiter=maxiter, disp=False)
        if best is None or float(result.llf) > float(best.llf):
            best = result
        if bool(result.converged):
            return result, method
    return best, "best_nonconverged"


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    data = mixed.load_data(args.data_dir)
    rows: list[dict[str, object]] = []
    status: list[dict[str, object]] = []

    for side in mixed.SIDES:
        for label in mixed.SIDE_LABELS[side]:
            for level in mixed.LEVELS:
                df = mixed.filter_group(data[side][level], "all")
                for focus_feature in mixed.FEATURES[side]:
                    terms = mixed.CONTROL_FEATURES[side][level] + [focus_feature]
                    try:
                        result, method = fit_result(df, label, terms, args.item_col, args.maxiter)
                    except Exception as exc:
                        status.append({
                            "side": side,
                            "label": label,
                            "level": level,
                            "focus_feature": focus_feature,
                            "status": f"failed:{type(exc).__name__}",
                            "message": str(exc)[:300],
                        })
                        continue

                    params = result.params
                    bse = result.bse
                    for predictor in terms:
                        coef = float(params.get(predictor, float("nan")))
                        se = float(bse.get(predictor, float("nan")))
                        rows.append({
                            "side": side,
                            "label": label,
                            "level": level,
                            "focus_feature": focus_feature,
                            "predictor": predictor,
                            "mean_coef": coef,
                            "ci95_halfwidth": 1.96 * se,
                            "n_folds": 1,
                            "n_rows": int(result.nobs),
                            "converged": bool(result.converged),
                            "optimizer": method,
                        })
                    status.append({
                        "side": side,
                        "label": label,
                        "level": level,
                        "focus_feature": focus_feature,
                        "status": "ok",
                        "message": "",
                        "converged": bool(result.converged),
                        "optimizer": method,
                        "llf": float(result.llf),
                        "n_rows": int(result.nobs),
                    })

    coef_df = pd.DataFrame(rows)
    status_df = pd.DataFrame(status)
    coef_df.to_csv(args.outdir / "figure2_coefficients.tsv", sep="\t", index=False)
    status_df.to_csv(args.outdir / "table_mixed_coefficients_fit_status.tsv", sep="\t", index=False)
    print(args.outdir / "figure2_coefficients.tsv")
    print(args.outdir / "table_mixed_coefficients_fit_status.tsv")


if __name__ == "__main__":
    main()
