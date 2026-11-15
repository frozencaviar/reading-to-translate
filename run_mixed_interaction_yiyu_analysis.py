#!/usr/bin/env python3
"""Mixed-effects interaction tests for task type and translation direction."""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2

import run_mixed_effects_yiyu_analysis as base


@dataclass(frozen=True)
class FitSummary:
    llf: float
    nobs: int
    converged: bool
    model_kind: str


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=script_dir / "attention-norm-yiyu-positive-dur")
    parser.add_argument("--outdir", type=Path, default=script_dir / "figures_yiyu_positive_dur_mixed")
    parser.add_argument("--item-col", default="sentence_item_id")
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--min-rows", type=int, default=40)
    return parser.parse_args()


def build_formula(label: str, controls: list[str], predictor: str, interaction: str | None) -> str:
    terms = controls + [predictor, "C(task_type)", "C(lang_pair)"]
    if interaction == "task":
        terms.append(f"{predictor}:C(task_type)")
    elif interaction == "direction":
        terms.append(f"{predictor}:C(lang_pair)")
    return f"{label} ~ " + " + ".join(terms)


def fit_formula(df: pd.DataFrame, formula: str, item_col: str, maxiter: int) -> FitSummary:
    work = df.copy()
    work["_all_group"] = "all"
    try:
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
                return FitSummary(float(result.llf), int(result.nobs), True, f"participant+item:{method}")
        return FitSummary(float(best.llf), int(best.nobs), False, "participant+item:best_nonconverged")
    except Exception:
        model = smf.mixedlm(formula, data=work, groups=work["translator_id"], re_formula="1")
        best = None
        for method in ("lbfgs", "bfgs", "cg"):
            result = model.fit(reml=False, method=method, maxiter=maxiter, disp=False)
            if best is None or float(result.llf) > float(best.llf):
                best = result
            if bool(result.converged):
                return FitSummary(float(result.llf), int(result.nobs), True, f"participant_only:{method}")
        return FitSummary(float(best.llf), int(best.nobs), False, "participant_only:best_nonconverged")


def fit_interaction_pair(
    df: pd.DataFrame,
    label: str,
    controls: list[str],
    predictor: str,
    interaction: str,
    item_col: str,
    maxiter: int,
    min_rows: int,
) -> dict[str, object]:
    required = list(dict.fromkeys(controls + [predictor, label, "task_type", "lang_pair", "translator_id", item_col]))
    work = df.dropna(subset=required).copy()
    if interaction == "task":
        work = work[work["task_type"].isin(["translating", "post-editing"])].copy()
        factor = "task_type"
    else:
        factor = "lang_pair"
    if len(work) < min_rows or work["translator_id"].nunique() < 2 or work[item_col].nunique() < 2 or work[factor].nunique() < 2:
        return {
            "status": "insufficient",
            "predictor": f"control+{predictor}*{interaction}",
            "feature": predictor,
            "n_rows": len(work),
            "n_participants": int(work["translator_id"].nunique()) if len(work) else 0,
            "n_items": int(work[item_col].nunique()) if len(work) else 0,
            "n_factor_levels": int(work[factor].nunique()) if len(work) else 0,
        }

    additive_formula = build_formula(label, controls, predictor, interaction=None)
    interaction_formula = build_formula(label, controls, predictor, interaction=interaction)
    try:
        additive = fit_formula(work, additive_formula, item_col, maxiter)
        full = fit_formula(work, interaction_formula, item_col, maxiter)
    except Exception as exc:
        return {
            "status": f"failed:{type(exc).__name__}",
            "message": str(exc)[:300],
            "predictor": f"control+{predictor}*{interaction}",
            "feature": predictor,
            "n_rows": len(work),
            "n_participants": int(work["translator_id"].nunique()),
            "n_items": int(work[item_col].nunique()),
            "n_factor_levels": int(work[factor].nunique()),
        }

    delta_ll = full.llf - additive.llf
    lr_stat = max(0.0, 2.0 * delta_ll)
    p_raw = float(chi2.sf(lr_stat, df=1))
    return {
        "status": "ok",
        "interaction": interaction,
        "predictor": f"control+{predictor}*{interaction}",
        "feature": predictor,
        "metric": float(delta_ll / max(full.nobs, 1)),
        "p_raw": p_raw,
        "n_participants": int(work["translator_id"].nunique()),
        "n_items": int(work[item_col].nunique()),
        "n_rows": int(full.nobs),
        "n_factor_levels": int(work[factor].nunique()),
        "llf_additive": additive.llf,
        "llf_interaction": full.llf,
        "lr_stat": lr_stat,
        "df_lrt": 1,
        "additive_model_kind": additive.model_kind,
        "interaction_model_kind": full.model_kind,
        "additive_converged": additive.converged,
        "interaction_converged": full.converged,
    }


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore")
    data = base.load_data(args.data_dir)
    records: list[dict[str, object]] = []
    started = time.time()
    for interaction in ["task", "direction"]:
        print(f"[mixed-interaction] {interaction}")
        for side in base.SIDES:
            for level in base.LEVELS:
                for label in base.SIDE_LABELS[side]:
                    df = base.filter_group(data[side][level], "all")
                    controls = base.CONTROL_FEATURES[side][level]
                    for predictor in base.FEATURES[side]:
                        rec = fit_interaction_pair(
                            df=df,
                            label=label,
                            controls=controls,
                            predictor=predictor,
                            interaction=interaction,
                            item_col=args.item_col,
                            maxiter=args.maxiter,
                            min_rows=args.min_rows,
                        )
                        records.append({**rec, "side": side, "label": label, "level": level})
        print(f"[mixed-interaction] finished {interaction} after {time.time() - started:.1f}s")

    table = pd.DataFrame(records)
    ok = table["status"].eq("ok")
    table["q_bh"] = float("nan")
    table.loc[ok, "q_bh"] = base.bh_adjust(table.loc[ok, "p_raw"])
    table["sig_bh"] = False
    table.loc[ok, "sig_bh"] = table.loc[ok, "q_bh"] < 0.05

    args.outdir.mkdir(parents=True, exist_ok=True)
    all_path = args.outdir / "table_mixed_interaction_all_tests.tsv"
    fig_path = args.outdir / "figure_interaction_mixed.tsv"
    table.to_csv(all_path, sep="\t", index=False)
    table[ok].to_csv(fig_path, sep="\t", index=False)
    print(all_path)
    print(fig_path)


if __name__ == "__main__":
    main()
