#!/usr/bin/env python3
"""Create mixed-effects Delta-LL tables compatible with top-journal plots.

Model comparison uses shared random-effects structures (Barr et al., 2013):
base and augmented models share the same vc_formula, differing only in
whether the focal predictor appears in the fixed effects. Random slopes
are uncorrelated by-participant slopes via vc_formula. Convergence
fallbacks are handled at the pair level to guarantee symmetric structure.
"""

from __future__ import annotations

import argparse
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2, zscore

try:
    from wordfreq import word_frequency

    FREQ_SOURCE = "wordfreq"
except Exception:  # pragma: no cover - depends on local environment
    word_frequency = None
    FREQ_SOURCE = "corpus_unigram_fallback"

warnings.filterwarnings("ignore")


SIDES = ["src", "tgt"]
LEVELS = ["token", "segment"]
SIDE_LABELS = {"src": ["log_TrtS"], "tgt": ["log_TrtT", "log_Dur"]}
DUR_MIN_MS = 20
MAIN_SUMMARY_GROUPS = [
    "all",
    "all_translating",
    "all_post-editing",
    "all_en-zh",
    "all_zh-en",
]
FULL_SPLIT_GROUPS = [
    "translating_en-zh",
    "translating_zh-en",
    "post-editing_en-zh",
    "post-editing_zh-en",
]
GROUP_FILTERS = {
    "all": {},
    "all_translating": {"task_type": "translating"},
    "all_post-editing": {"task_type": "post-editing"},
    "all_en-zh": {"lang_pair": "en-zh"},
    "all_zh-en": {"lang_pair": "zh-en"},
    "translating_en-zh": {"lang_pair": "en-zh", "task_type": "translating"},
    "translating_zh-en": {"lang_pair": "zh-en", "task_type": "translating"},
    "post-editing_en-zh": {"lang_pair": "en-zh", "task_type": "post-editing"},
    "post-editing_zh-en": {"lang_pair": "zh-en", "task_type": "post-editing"},
}
FEATURES = {
    "src": ["src_surp", "x2x", "x2cont", "x2eos", "x_ent", "cont2x", "tgtseq2x"],
    "tgt": ["tgt_surp", "xent", "eos_xatt", "y2srcseq_ent", "y2y", "y2cont", "y_ent"],
}
CONTROL_FEATURES = {
    side: {level: [f"{side}_seg_len", f"{side}_freq", f"{side}_posq"] for level in LEVELS}
    for side in SIDES
}


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
    parser.add_argument(
        "--groups",
        default=",".join(MAIN_SUMMARY_GROUPS),
        help="Comma-separated group names. Use 'main' or 'all' as shortcuts.",
    )
    parser.add_argument(
        "--tables",
        default="figure5,figure3",
        help="Comma-separated table families to create: figure5, figure3.",
    )
    parser.add_argument(
        "--item-col",
        default="sentence_item_id",
        help="Random item intercept column. Use item_id for source-unit items.",
    )
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--min-rows", type=int, default=40)
    return parser.parse_args()


def infer_task_type(session_id: object) -> str:
    parts = str(session_id).split("_")
    if len(parts) > 1 and parts[1]:
        return {"T": "translating", "P": "post-editing", "C": "copying"}.get(parts[1][0].upper(), "other")
    return "other"


def infer_task_code(session_id: object) -> str:
    parts = str(session_id).split("_")
    return parts[1] if len(parts) > 1 and parts[1] else "unknown"


def zscore_col(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if np.isfinite(arr).sum() < 2:
        return pd.Series(arr, index=values.index)
    return pd.Series(zscore(arr, nan_policy="omit"), index=values.index)


def corpus_frequency_maps(frames: list[pd.DataFrame]) -> dict[str, dict[str, float]]:
    counts: dict[str, dict[str, int]] = {}
    totals: dict[str, int] = {}
    for frame in frames:
        for side in SIDES:
            token_col = f"{side}_tokens"
            lang_col = f"{side}_lang"
            if token_col not in frame.columns or lang_col not in frame.columns:
                continue
            for tokens, lang in frame[[token_col, lang_col]].dropna().itertuples(index=False):
                lang = str(lang)
                counts.setdefault(lang, {})
                for token in str(tokens).split("_"):
                    if not token or token == "---":
                        continue
                    counts[lang][token] = counts[lang].get(token, 0) + 1
                    totals[lang] = totals.get(lang, 0) + 1
    freqs: dict[str, dict[str, float]] = {}
    for lang, lang_counts in counts.items():
        denom = max(totals.get(lang, 0), 1)
        freqs[lang] = {token: count / denom for token, count in lang_counts.items()}
    return freqs


def mean_log_frequency(tokens: object, lang: object, fallback: dict[str, dict[str, float]]) -> float:
    pieces = [w for w in str(tokens).split("_") if w and w != "---"]
    if not pieces:
        return np.nan
    lang = str(lang)
    if word_frequency is not None:
        vals = [max(word_frequency(piece, lang), 1e-8) for piece in pieces]
    else:
        lang_freqs = fallback.get(lang, {})
        vals = [max(lang_freqs.get(piece, 1e-8), 1e-8) for piece in pieces]
    return float(np.log(np.mean(vals)))


def preprocess(df: pd.DataFrame, fallback_freqs: dict[str, dict[str, float]]) -> pd.DataFrame:
    df = df.copy()
    for label in ["TrtS", "Dur", "TrtT"]:
        if label in df.columns:
            df[label] = pd.to_numeric(df[label], errors="coerce")

    df["log_TrtS"] = np.where(df["TrtS"] >= DUR_MIN_MS, np.log(df["TrtS"]), np.nan)
    df["log_TrtT"] = np.where(df["TrtT"] >= DUR_MIN_MS, np.log(df["TrtT"]), np.nan)
    df["log_Dur"] = np.where((df["Dur"] > 0) & (df["Dur"] >= DUR_MIN_MS), np.log(df["Dur"]), np.nan)
    for label in ["log_TrtS", "log_TrtT", "log_Dur"]:
        df[label] = zscore_col(df[label])

    for side in SIDES:
        df[f"{side}_tokens"] = df[f"{side}_tokens"].astype(str)
        df[f"{side}_freq"] = df[[f"{side}_tokens", f"{side}_lang"]].apply(
            lambda x: mean_log_frequency(x.iloc[0], x.iloc[1], fallback_freqs),
            axis=1,
        )

    df["lang_pair"] = df[["src_lang", "tgt_lang"]].apply(lambda x: "-".join(x.astype(str)), axis=1)
    df["task_type"] = df["session_id"].apply(infer_task_type)
    df["task_code"] = df["session_id"].apply(infer_task_code)
    df["translator_id"] = df["session_id"].apply(lambda s: str(s).split("_")[0])
    df["material_id"] = df[["study", "task_code"]].apply(lambda x: f"{x.iloc[0]}::{x.iloc[1]}", axis=1)
    df["sentence_item_id"] = df[["material_id", "src_segid"]].apply(lambda x: f"{x.iloc[0]}::s{x.iloc[1]}", axis=1)
    df["src_unit_item_id"] = df[["sentence_item_id", "src_token_ids"]].apply(
        lambda x: f"{x.iloc[0]}::src_{x.iloc[1]}",
        axis=1,
    )
    df["tgt_position_item_id"] = df[["sentence_item_id", "tgt_token_ids"]].apply(
        lambda x: f"{x.iloc[0]}::tgt_{x.iloc[1]}",
        axis=1,
    )
    df["item_id"] = df["src_unit_item_id"]
    df["session_sentence_id"] = df[["study", "session_id", "src_segid"]].apply(
        lambda x: f"{x.iloc[0]}_{x.iloc[1]}_{x.iloc[2]}",
        axis=1,
    )
    df["sentence_id"] = df["session_sentence_id"]
    df["src_seg_len"] = df["src_token_ids"].apply(lambda ids: len(str(ids).split("+")) if pd.notnull(ids) else np.nan)
    df["tgt_seg_len"] = df["tgt_token_ids"].apply(lambda ids: len(str(ids).split("+")) if pd.notnull(ids) else np.nan)
    if "y_ent" in df.columns:
        df["y_ent"] = df["y_ent"].fillna(0)

    for col in [
        "src_posq",
        "tgt_posq",
        "xent",
        "xmi",
        "x2cont",
        "cont2x",
        "x_ent",
        "x2eos",
        "tgtseq2x",
        "src_surp",
        "y2srcseq_ent",
        "eos_xatt",
        "y2cont",
        "y_ent",
        "tgt_surp",
        "src_freq",
        "tgt_freq",
        "x2x",
        "y2y",
    ]:
        if col in df.columns:
            df[col] = zscore_col(df[col])
    for c in df.columns:
        if isinstance(df[c].dtype, pd.StringDtype):
            df[c] = df[c].astype("object")
    return df


def load_data(data_dir: Path) -> dict[str, dict[str, pd.DataFrame]]:
    tok_seg_raw = pd.read_csv(data_dir / "yiyu-features-token-segment.tsv", sep="\t")
    seg_tok_raw = pd.read_csv(data_dir / "yiyu-features-segment-token.tsv", sep="\t")
    sent_raw = pd.read_csv(data_dir / "yiyu-features-sentence-sentence.tsv", sep="\t")
    fallback_freqs = corpus_frequency_maps([tok_seg_raw, seg_tok_raw, sent_raw])
    tok_seg = preprocess(tok_seg_raw, fallback_freqs)
    seg_tok = preprocess(seg_tok_raw, fallback_freqs)
    sent = preprocess(sent_raw, fallback_freqs)
    return {
        "src": {
            "token": tok_seg,
            "segment": seg_tok.drop_duplicates(["study", "session_id", "src_segid", "src_token_ids"]).copy(),
            "sentence": sent,
        },
        "tgt": {
            "token": seg_tok,
            "segment": tok_seg.drop_duplicates(["study", "session_id", "src_segid", "tgt_token_ids"]).copy(),
            "sentence": sent,
        },
    }


def filter_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
    out = df
    for col, value in GROUP_FILTERS[group].items():
        out = out[out[col] == value]
    return out.copy()


def fixed_terms(features: list[str], include_lang_pair: bool = True) -> list[str]:
    terms = list(features)
    if include_lang_pair:
        terms.append("C(lang_pair)")
    return terms


def build_formula(label: str, features: list[str], include_lang_pair: bool = True) -> str:
    return f"{label} ~ " + " + ".join(fixed_terms(features, include_lang_pair=include_lang_pair))


def fit_mixed_model(
    df: pd.DataFrame,
    label: str,
    features: list[str],
    item_col: str,
    maxiter: int,
    vc_formula: dict[str, str] | None = None,
) -> FitSummary:
    """Fit mixed model. If vc_formula is provided, use it (may include random slopes).
    Otherwise use default crossed RI-only vc_formula."""
    formula = build_formula(label, features)
    work = df.copy()
    work["_all_group"] = "all"

    if vc_formula is None:
        vc_formula = {"participant": "0 + C(translator_id)", "item": f"0 + C({item_col})"}

    try:
        model = smf.mixedlm(formula, data=work, groups=work["_all_group"], re_formula="0", vc_formula=vc_formula)
        best_result = None
        for method in ("lbfgs", "bfgs", "cg"):
            result = model.fit(reml=False, method=method, maxiter=maxiter, disp=False)
            if best_result is None or float(result.llf) > float(best_result.llf):
                best_result = result
            if bool(result.converged):
                return FitSummary(float(result.llf), int(result.nobs), True, f"participant+item:{method}")
        return FitSummary(float(best_result.llf), int(best_result.nobs), False, "participant+item:best_nonconverged")
    except Exception:
        return FitSummary(float("nan"), 0, False, "failed")


def fit_pair(
    df: pd.DataFrame,
    label: str,
    base_features: list[str],
    predictor: str,
    item_col: str,
    maxiter: int,
    min_rows: int,
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    required = list(dict.fromkeys(base_features + [predictor, label, "lang_pair", "translator_id", item_col]))
    work = df.dropna(subset=required).copy()
    status_rows: list[dict[str, object]] = []
    if len(work) < min_rows or work["translator_id"].nunique() < 2 or work[item_col].nunique() < 2:
        return None, [{
            "predictor": predictor,
            "status": "insufficient",
            "n_rows": len(work),
            "n_participants": int(work["translator_id"].nunique()) if len(work) else 0,
            "n_items": int(work[item_col].nunique()) if len(work) else 0,
        }]

    try:
        # Tier 1: shared random slopes (Barr et al., 2013)
        vc_shared = {
            "participant_intercept": "0 + C(translator_id)",
            "participant_slope": f"0 + C(translator_id):{predictor}",
            "item": f"0 + C({item_col})",
        }
        base = fit_mixed_model(work, label, base_features, item_col, maxiter, vc_formula=vc_shared)
        pred = fit_mixed_model(work, label, base_features + [predictor], item_col, maxiter, vc_formula=vc_shared)
        tier = "random_slope"
        if not (base.converged and pred.converged):
            # Tier 2: RI-only fallback — must be symmetric
            base = fit_mixed_model(work, label, base_features, item_col, maxiter)
            pred = fit_mixed_model(work, label, base_features + [predictor], item_col, maxiter)
            tier = "RI-only" if (base.converged and pred.converged) else "RI-only_nonconverged"
    except Exception as exc:
        return None, [{
            "predictor": predictor,
            "status": f"failed:{type(exc).__name__}",
            "message": str(exc)[:300],
            "n_rows": len(work),
            "n_participants": int(work["translator_id"].nunique()),
            "n_items": int(work[item_col].nunique()),
        }]

    if not (base.converged and pred.converged):
        return None, [{
            "predictor": predictor,
            "status": tier,
            "message": "paired models did not converge after RI-only fallback",
            "n_rows": len(work),
            "n_participants": int(work["translator_id"].nunique()),
            "n_items": int(work[item_col].nunique()),
        }]

    delta_ll = pred.llf - base.llf
    lr_stat = max(0.0, 2.0 * delta_ll)
    p_raw = float(chi2.sf(lr_stat, df=1))
    record = {
        "predictor": f"control+{predictor}",
        "feature": predictor,
        "metric": float(delta_ll / max(pred.nobs, 1)),
        "p_raw": p_raw,
        "n_participants": int(work["translator_id"].nunique()),
        "n_items": int(work[item_col].nunique()),
        "n_rows": int(pred.nobs),
        "llf_control": base.llf,
        "llf_predictor": pred.llf,
        "lr_stat": lr_stat,
        "base_model_kind": base.model_kind,
        "predictor_model_kind": pred.model_kind,
        "base_converged": base.converged,
        "predictor_converged": pred.converged,
        "tier": tier,
    }
    return record, []


def bh_adjust(pvalues: pd.Series) -> pd.Series:
    pvals = pd.to_numeric(pvalues, errors="coerce").to_numpy(dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)
    valid = np.isfinite(pvals)
    if not valid.any():
        return pd.Series(out, index=pvalues.index)
    valid_p = pvals[valid]
    order = np.argsort(valid_p)
    sorted_p = valid_p[order]
    adjusted = np.empty_like(sorted_p)
    running = 1.0
    n = len(sorted_p)
    for idx in range(n - 1, -1, -1):
        running = min(running, sorted_p[idx] * n / (idx + 1))
        adjusted[idx] = min(running, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[np.where(valid)[0]] = restored
    return pd.Series(out, index=pvalues.index)


def group_names(spec: str) -> list[str]:
    if spec == "main":
        return MAIN_SUMMARY_GROUPS
    if spec == "all":
        return MAIN_SUMMARY_GROUPS + FULL_SPLIT_GROUPS
    groups = [g.strip() for g in spec.split(",") if g.strip()]
    unknown = [g for g in groups if g not in GROUP_FILTERS]
    if unknown:
        raise ValueError(f"Unknown group(s): {', '.join(unknown)}")
    return groups


def run_tables(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = load_data(args.data_dir)
    groups = group_names(args.groups)
    requested = {item.strip() for item in args.tables.split(",") if item.strip()}
    records: list[dict[str, object]] = []
    status: list[dict[str, object]] = []
    start = time.time()

    for group in groups:
        print(f"[mixed] group={group}")
        for side in SIDES:
            for level in LEVELS:
                for label in SIDE_LABELS[side]:
                    df = filter_group(data[side][level], group)
                    base_features = CONTROL_FEATURES[side][level]
                    predictors = FEATURES[side]
                    if "figure5" in requested:
                        for predictor in predictors:
                            rec, problems = fit_pair(
                                df, label, base_features, predictor, args.item_col, args.maxiter, args.min_rows
                            )
                            status.extend({**p, "group": group, "side": side, "level": level, "label": label, "delta": "*ctrl"} for p in problems)
                            if rec is not None:
                                records.append({**rec, "group": group, "side": side, "label": label, "level": level, "delta": "*ctrl"})

                    if "figure3" in requested:
                        surp_base = base_features + (["src_surp"] if side == "src" else ["xent"])
                        attention_predictors = [
                            p for p in predictors
                            if p not in (["src_surp"] if side == "src" else ["tgt_surp", "xent"])
                        ]
                        for predictor in attention_predictors:
                            rec, problems = fit_pair(
                                df, label, surp_base, predictor, args.item_col, args.maxiter, args.min_rows
                            )
                            status.extend({**p, "group": group, "side": side, "level": level, "label": label, "delta": "*ctrl+surp"} for p in problems)
                            if rec is not None:
                                records.append({**rec, "group": group, "side": side, "label": label, "level": level, "delta": "*ctrl+surp"})
        elapsed = time.time() - start
        print(f"[mixed] finished {group} after {elapsed:.1f}s")

    table = pd.DataFrame(records)
    if not table.empty:
        table["q_bh"] = bh_adjust(table["p_raw"])
        table["sig_bh"] = table["q_bh"] < 0.05
    status_columns = [
        "group",
        "side",
        "level",
        "label",
        "delta",
        "predictor",
        "status",
        "message",
        "n_rows",
        "n_participants",
        "n_items",
    ]
    status_table = pd.DataFrame(status, columns=status_columns)
    metadata = pd.DataFrame([
        {"key": "model", "value": "MixedLM ML, crossed random effects with by-participant uncorrelated random slopes (Barr et al., 2013); convergence fallback to RI-only if random slopes fail"},
        {"key": "item_col", "value": args.item_col},
        {"key": "frequency_source", "value": FREQ_SOURCE},
        {"key": "metric", "value": "(llf_control_plus_predictor - llf_control) / nobs"},
        {"key": "p_raw", "value": "likelihood-ratio chi-square test, df=1"},
        {"key": "groups", "value": ",".join(groups)},
        {"key": "tables", "value": ",".join(sorted(requested))},
    ])
    return table, status_table, metadata


def write_outputs(table: pd.DataFrame, status: pd.DataFrame, metadata: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if table.empty:
        raise RuntimeError("No mixed-effects rows were produced.")

    base_cols = ["group", "side", "label", "level", "predictor", "feature", "metric", "p_raw", "n_participants", "q_bh", "sig_bh", "tier", "base_converged", "predictor_converged"]
    fig5 = table[table["delta"] == "*ctrl"].copy()
    fig3 = table[table["delta"].isin(["*ctrl", "*ctrl+surp"])].copy()

    fig5_main = fig5[fig5["group"].isin(MAIN_SUMMARY_GROUPS)][base_cols].copy()
    fig5_split = fig5[fig5["group"].isin(FULL_SPLIT_GROUPS)][base_cols].copy()
    fig3_cols = ["group", "side", "label", "level", "delta", "predictor", "feature", "metric", "p_raw", "n_participants", "q_bh", "sig_bh", "tier", "base_converged", "predictor_converged"]
    fig3_main = fig3[fig3["group"].isin(MAIN_SUMMARY_GROUPS)][fig3_cols].copy()
    fig3_split = fig3[fig3["group"].isin(FULL_SPLIT_GROUPS)][fig3_cols].copy()
    fig1 = fig5[fig5["group"].eq("all")][base_cols[1:]].copy()

    fig1.to_csv(outdir / "figure1_delta_llh_main.tsv", sep="\t", index=False)
    fig5_main.to_csv(outdir / "figure5_delta_llh_grouped.tsv", sep="\t", index=False)
    fig5_split.to_csv(outdir / "figure5_appendix_delta_llh_grouped_split.tsv", sep="\t", index=False)
    fig3_main.to_csv(outdir / "figure3_delta_llh_att_grouped.tsv", sep="\t", index=False)
    fig3_split.to_csv(outdir / "figure3_appendix_delta_llh_att_grouped_split.tsv", sep="\t", index=False)
    table.to_csv(outdir / "table_mixed_effects_all_tests.tsv", sep="\t", index=False)
    status.to_csv(outdir / "table_mixed_effects_fit_status.tsv", sep="\t", index=False)
    metadata.to_csv(outdir / "table_mixed_effects_metadata.tsv", sep="\t", index=False)


def main() -> None:
    args = parse_args()
    table, status, metadata = run_tables(args)
    write_outputs(table, status, metadata, args.outdir)
    print(f"[mixed] wrote mixed-effects tables to {args.outdir}")


if __name__ == "__main__":
    main()
