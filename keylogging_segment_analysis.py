"""
Segment-level keylogging extension for the YiyuZhang_FIX positive-Dur run.

This script keeps the current NMT feature extraction fixed and adds richer
TPR-DB .sg keylogging outcomes at the sentence/segment level.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import zscore
from wordfreq import word_frequency

warnings.filterwarnings("ignore")


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURE_DIR = SCRIPT_DIR / "attention-norm-yiyu-positive-dur"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "figures_yiyu_keylogging"

DUR_MIN_MS = 20
MAIN_OUTCOMES = ["log_PreGap", "log_TD1000", "log1p_TB1000", "log1p_Del"]
ROBUSTNESS_OUTCOMES = ["log1p_Nedit"]
SANITY_OUTCOMES = ["log_Dur_sg"]
PREDICTORS = ["tgt_surp", "xent", "eos_xatt", "y2srcseq_ent", "y2y", "y2cont", "y_ent"]
CONTROLS = ["tgt_seg_len", "tgt_freq", "tgt_posq"]
MODEL_COLUMNS = CONTROLS + PREDICTORS
GROUPS = {
    "all": {},
    "all_translating": {"task_type": "translating"},
    "all_post-editing": {"task_type": "post-editing"},
    "all_en-zh": {"lang_pair": "en-zh"},
    "all_zh-en": {"lang_pair": "zh-en"},
    "translating_en-zh": {"task_type": "translating", "lang_pair": "en-zh"},
    "translating_zh-en": {"task_type": "translating", "lang_pair": "zh-en"},
    "post-editing_en-zh": {"task_type": "post-editing", "lang_pair": "en-zh"},
    "post-editing_zh-en": {"task_type": "post-editing", "lang_pair": "zh-en"},
}

FEATURE_LABELS = {
    "tgt_surp": r"$S_{\mathrm{lm}}(v)$",
    "xent": r"$S_{\mathrm{mt}}(v)$",
    "eos_xatt": r"$f^c_{v,\mathrm{eos}}$",
    "y2srcseq_ent": r"$H^c_{v,x}$",
    "y2y": r"$f^d_{v,v}$",
    "y2cont": r"$f^d_{v,\bar{v}}$",
    "y_ent": r"$H^d_{v,\bar{v}}$",
}
OUTCOME_LABELS = {
    "log_PreGap": "Planning pause",
    "log_TD1000": "Typing duration",
    "log1p_TB1000": "Typing bursts",
    "log1p_Del": "Deletion effort",
    "log1p_Nedit": "Edit count",
    "log_Dur_sg": "Total duration",
}
GROUP_LABELS = {
    "all": "All",
    "all_translating": "All HT",
    "all_post-editing": "All PE",
    "all_en-zh": "All en->zh",
    "all_zh-en": "All zh->en",
    "translating_en-zh": "HT en->zh",
    "translating_zh-en": "HT zh->en",
    "post-editing_en-zh": "PE en->zh",
    "post-editing_zh-en": "PE zh->en",
}


@dataclass
class FitResult:
    llh: pd.DataFrame
    params: pd.Series


def infer_task_type(session_id: str) -> str:
    parts = str(session_id).split("_")
    if len(parts) > 1 and parts[1]:
        return {"T": "translating", "P": "post-editing", "C": "copying"}.get(parts[1][0].upper(), "other")
    return "other"


def infer_task_code(session_id: str) -> str:
    parts = str(session_id).split("_")
    return parts[1] if len(parts) > 1 and parts[1] else "unknown"


def mean_log_word_frequency(tokens: object, lang: str) -> float:
    words = [w for w in str(tokens).split("_") if w and w != "---"]
    if not words:
        return np.nan
    vals = [max(word_frequency(w, lang), 1e-8) for w in words]
    return float(np.log(np.mean(vals)))


def parse_segment_id(value: object) -> float:
    text = str(value)
    if "+" in text:
        return np.nan
    try:
        return float(int(text) - 1)
    except Exception:
        return np.nan


def zscore_series(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if np.isfinite(arr).sum() < 2:
        return pd.Series(arr, index=values.index)
    return pd.Series(zscore(arr, nan_policy="omit"), index=values.index)


def likelihood(preds: np.ndarray, labels: np.ndarray, std: float) -> np.ndarray:
    std = max(float(std), 1e-10)
    return -0.5 * np.log(2 * math.pi * std**2) - ((preds - labels) ** 2) / (2 * std**2)


def benjamini_hochberg(pvalues: list[float]) -> np.ndarray:
    pvals = np.asarray(pvalues, dtype=float)
    if pvals.size == 0:
        return np.array([], dtype=float)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    adjusted = np.empty_like(sorted_p)
    running = 1.0
    n = len(sorted_p)
    for i in range(n - 1, -1, -1):
        running = min(running, sorted_p[i] * n / (i + 1))
        adjusted[i] = min(running, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def signflip_pvalue(values: np.ndarray, n_resamples: int = 1000, seed: int = 42) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    observed = abs(float(np.mean(vals)))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, vals.size))
    permuted = np.abs((signs * vals).mean(axis=1))
    return float((np.sum(permuted >= observed) + 1) / (n_resamples + 1))


def load_sg_tables(table_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(table_dir / "*.sg")))
    if not files:
        raise FileNotFoundError(f"No .sg files found in {table_dir}")

    keep_cols = [
        "Study", "Session", "SL", "TL", "Task", "Text", "Part", "STseg", "TTseg",
        "Nedit", "Dur", "FDur", "PreGap", "TG1000", "TD1000", "TB1000", "Del", "Ins",
        "ParTrtS", "ParTrtT", "TokT", "LenT",
    ]
    frames = []
    for file in files:
        df = pd.read_csv(file, sep="\t")
        cols = [c for c in keep_cols if c in df.columns]
        frames.append(df[cols].copy())
    sg = pd.concat(frames, ignore_index=True)
    sg = sg.rename(columns={
        "Study": "study",
        "Session": "session_id",
        "SL": "src_lang_sg",
        "TL": "tgt_lang_sg",
        "Task": "task_code",
        "Part": "part_id_sg",
    })
    sg["src_segid"] = sg["STseg"].apply(parse_segment_id)
    sg["tgt_segid"] = sg["TTseg"].apply(parse_segment_id)
    sg = sg.dropna(subset=["src_segid", "tgt_segid"])
    sg[["src_segid", "tgt_segid"]] = sg[["src_segid", "tgt_segid"]].astype(int)
    for col in ["Nedit", "Dur", "FDur", "PreGap", "TG1000", "TD1000", "TB1000", "Del", "Ins", "ParTrtS", "ParTrtT"]:
        if col in sg.columns:
            sg[col] = pd.to_numeric(sg[col], errors="coerce")
    return sg


def load_feature_segments(feature_dir: Path) -> pd.DataFrame:
    sent_path = feature_dir / "yiyu-features-sentence-sentence.tsv"
    if not sent_path.exists():
        raise FileNotFoundError(f"Missing current sentence-level feature file: {sent_path}")
    feats = pd.read_csv(sent_path, sep="\t")
    feats = feats.copy()
    feats["src_segid"] = pd.to_numeric(feats["src_segid"], errors="coerce").astype("Int64")
    feats["tgt_segid"] = pd.to_numeric(feats["tgt_segid"], errors="coerce").astype("Int64")
    feats = feats.dropna(subset=["src_segid", "tgt_segid"])
    feats[["src_segid", "tgt_segid"]] = feats[["src_segid", "tgt_segid"]].astype(int)
    return feats


def build_dataset(feature_dir: Path, table_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = load_feature_segments(feature_dir)
    sg = load_sg_tables(table_dir)
    merge_keys = ["study", "session_id", "src_segid", "tgt_segid"]
    merged = features.merge(sg, on=merge_keys, how="inner", validate="one_to_one", suffixes=("", "_sg"))
    merged = merged.copy()
    merged["row_uid"] = np.arange(len(merged), dtype=int)
    merged["lang_pair"] = merged[["src_lang", "tgt_lang"]].apply(lambda x: "-".join(x.astype(str)), axis=1)
    merged["task_type"] = merged["session_id"].apply(infer_task_type)
    merged["task_code"] = merged["session_id"].apply(infer_task_code)
    merged["translator_id"] = merged["session_id"].apply(lambda s: str(s).split("_")[0])
    merged["material_id"] = merged[["study", "task_code"]].apply(lambda x: f"{x.iloc[0]}::{x.iloc[1]}", axis=1)
    merged["sentence_item_id"] = merged[["material_id", "src_segid"]].apply(
        lambda x: f"{x.iloc[0]}::s{x.iloc[1]}", axis=1
    )
    merged["src_unit_item_id"] = merged[["sentence_item_id", "src_token_ids"]].apply(
        lambda x: f"{x.iloc[0]}::src_{x.iloc[1]}", axis=1
    )
    merged["tgt_position_item_id"] = merged[["sentence_item_id", "tgt_token_ids"]].apply(
        lambda x: f"{x.iloc[0]}::tgt_{x.iloc[1]}", axis=1
    )
    merged["item_id"] = merged["sentence_item_id"]
    merged["session_sentence_id"] = merged[merge_keys].astype(str).agg("_".join, axis=1)
    merged["sentence_id"] = merged[merge_keys].astype(str).agg("_".join, axis=1)
    merged["tgt_seg_len"] = merged["tgt_token_ids"].apply(lambda ids: len(str(ids).split("+")) if pd.notnull(ids) else np.nan)
    merged["tgt_freq"] = merged[["tgt_tokens", "tgt_lang"]].apply(lambda x: mean_log_word_frequency(x.iloc[0], x.iloc[1]), axis=1)

    sg_dur_col = "Dur_sg" if "Dur_sg" in merged.columns else "Dur"
    produced = pd.to_numeric(merged[sg_dur_col], errors="coerce") >= DUR_MIN_MS
    merged["log_PreGap"] = np.where(pd.to_numeric(merged["PreGap"], errors="coerce") >= DUR_MIN_MS, np.log(merged["PreGap"]), np.nan)
    merged["log_TD1000"] = np.where(pd.to_numeric(merged["TD1000"], errors="coerce") >= DUR_MIN_MS, np.log(merged["TD1000"]), np.nan)
    merged["log1p_TB1000"] = np.where(produced & merged["TB1000"].notnull(), np.log1p(merged["TB1000"]), np.nan)
    merged["log1p_Del"] = np.where(merged["Del"].notnull(), np.log1p(merged["Del"].clip(lower=0)), np.nan)
    merged["log1p_Nedit"] = np.where(merged["Nedit"].notnull(), np.log1p(merged["Nedit"].clip(lower=0)), np.nan)
    merged["log_Dur_sg"] = np.where(produced, np.log(merged[sg_dur_col]), np.nan)

    for col in MAIN_OUTCOMES + ROBUSTNESS_OUTCOMES + SANITY_OUTCOMES + MODEL_COLUMNS:
        if col in merged.columns:
            merged[col] = zscore_series(merged[col])

    coverage = build_coverage_table(features, sg, merged)
    return merged, coverage


def build_coverage_table(features: pd.DataFrame, sg: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"check": "feature_rows", "value": len(features)},
        {"check": "sg_rows", "value": len(sg)},
        {"check": "merged_rows", "value": len(merged)},
        {"check": "feature_sessions", "value": features["session_id"].nunique()},
        {"check": "sg_sessions", "value": sg["session_id"].nunique()},
        {"check": "merged_sessions", "value": merged["session_id"].nunique()},
        {"check": "merged_translators", "value": merged["translator_id"].nunique()},
        {"check": "merged_unique_segment_keys", "value": merged[["study", "session_id", "src_segid", "tgt_segid"]].drop_duplicates().shape[0]},
    ]
    for raw in ["PreGap", "TD1000", "TB1000", "Del", "Nedit", "ParTrtS", "ParTrtT"]:
        if raw in merged.columns:
            vals = pd.to_numeric(merged[raw], errors="coerce")
            rows.append({"check": f"{raw}_nonmissing", "value": int(vals.notnull().sum())})
            rows.append({"check": f"{raw}_positive", "value": int((vals > 0).sum())})
            rows.append({"check": f"{raw}_max", "value": float(vals.max())})
    return pd.DataFrame(rows)


def preprocessing_summary(df: pd.DataFrame, outcomes: list[str]) -> pd.DataFrame:
    rows = []
    for outcome in outcomes:
        valid = df[outcome].notnull()
        rows.append({
            "outcome": outcome,
            "outcome_label": OUTCOME_LABELS.get(outcome, outcome),
            "total_rows": int(len(df)),
            "valid_rows": int(valid.sum()),
            "dropped_rows": int((~valid).sum()),
            "valid_rate": float(valid.mean()),
            "unique_sessions": int(df.loc[valid, "session_id"].nunique()),
            "unique_translators": int(df.loc[valid, "translator_id"].nunique()),
            "n_translating": int((df.loc[valid, "task_type"] == "translating").sum()),
            "n_post_editing": int((df.loc[valid, "task_type"] == "post-editing").sum()),
        })
    return pd.DataFrame(rows)


def fit_model(train: pd.DataFrame, test: pd.DataFrame, outcome: str, features: list[str]) -> FitResult | None:
    required = features + ["lang_pair", outcome]
    train = train.dropna(subset=required).copy()
    test = test.dropna(subset=required).copy()
    if len(train) < 20 or len(test) < 5:
        return None
    formula = f"{outcome} ~ " + " + ".join(features + ["C(lang_pair)"])
    try:
        model = smf.ols(formula, data=train).fit()
        train_pred = np.asarray(model.predict(train), dtype=float)
        test_pred = np.asarray(model.predict(test), dtype=float)
    except Exception:
        return None
    std = np.std(train_pred - train[outcome].to_numpy(dtype=float))
    llh = likelihood(test_pred, test[outcome].to_numpy(dtype=float), std)
    meta = test[["row_uid", "translator_id", "sentence_id", "session_id", "task_type", "lang_pair"]].copy()
    meta["llh"] = llh
    return FitResult(llh=meta, params=model.params)


def split_by_sentence(df: pd.DataFrame, n_splits: int, seed: int) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    sentences = np.array(sorted(df["sentence_id"].dropna().unique()))
    if len(sentences) < 2:
        return []
    rng = np.random.default_rng(seed)
    rng.shuffle(sentences)
    folds = np.array_split(sentences, min(n_splits, len(sentences)))
    splits = []
    for fold in folds:
        fold_set = set(fold.tolist())
        train = df[~df["sentence_id"].isin(fold_set)].copy()
        test = df[df["sentence_id"].isin(fold_set)].copy()
        if len(train) and len(test):
            splits.append((train, test))
    return splits


def filter_group(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    out = df
    for col, value in GROUPS[group_name].items():
        out = out[out[col] == value]
    return out.copy()


def run_cv(df: pd.DataFrame, group_name: str, outcomes: list[str], n_splits: int, seed: int) -> pd.DataFrame:
    group_df = filter_group(df, group_name)
    records = []
    for outcome in outcomes:
        outcome_df = group_df[group_df[outcome].notnull()].copy()
        splits = split_by_sentence(outcome_df, n_splits=n_splits, seed=seed)
        if not splits:
            continue
        control_by_fold = []
        for train, test in splits:
            control_by_fold.append(fit_model(train, test, outcome, CONTROLS))
        for predictor in PREDICTORS:
            deltas = []
            for (train, test), control in zip(splits, control_by_fold):
                if control is None:
                    continue
                pred = fit_model(train, test, outcome, CONTROLS + [predictor])
                if pred is None:
                    continue
                merged = pred.llh[["row_uid", "llh", "translator_id", "sentence_id", "session_id", "task_type", "lang_pair"]].merge(
                    control.llh[["row_uid", "llh"]].rename(columns={"llh": "llh_control"}),
                    on="row_uid",
                    how="inner",
                )
                merged["delta"] = merged["llh"] - merged["llh_control"]
                deltas.append(merged)
            if not deltas:
                continue
            delta_df = pd.concat(deltas, ignore_index=True)
            part_means = delta_df.groupby("translator_id", as_index=False)["delta"].mean()
            p_raw = signflip_pvalue(part_means["delta"].to_numpy(), seed=seed)
            records.append({
                "group": group_name,
                "outcome": outcome,
                "outcome_label": OUTCOME_LABELS.get(outcome, outcome),
                "predictor": f"control+{predictor}",
                "feature": predictor,
                "feature_label": FEATURE_LABELS.get(predictor, predictor),
                "metric": float(delta_df["delta"].mean()),
                "p_raw": p_raw,
                "n_participants": int(part_means["translator_id"].nunique()),
                "n_rows": int(len(delta_df)),
                "n_splits": int(len(splits)),
            })
    result = pd.DataFrame(records)
    if not result.empty:
        result["q_bh"] = benjamini_hochberg(result["p_raw"].fillna(1.0).tolist())
        result["sig_bh"] = result["q_bh"] < 0.05
    return result


def plot_main_delta(table: pd.DataFrame, output_dir: Path) -> None:
    main = table[table["outcome"].isin(MAIN_OUTCOMES)].copy()
    if main.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    axes = axes.ravel()
    for ax, outcome in zip(axes, MAIN_OUTCOMES):
        sub = main[main["outcome"] == outcome].copy()
        sub["feature"] = pd.Categorical(sub["feature"], PREDICTORS, ordered=True)
        sub = sub.sort_values("feature")
        x = np.arange(len(sub))
        colors = np.where(sub["sig_bh"], "#1f77b4", "#b8c2cc")
        ax.axhline(0, color="#9ca3af", lw=0.8)
        ax.scatter(x, sub["metric"], s=70, c=colors, edgecolor="#1f2937", linewidth=0.5, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([FEATURE_LABELS.get(f, f) for f in sub["feature"]], rotation=35, ha="right", fontsize=9)
        ax.set_title(OUTCOME_LABELS[outcome], fontsize=12)
        ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    axes[0].set_ylabel(r"Held-out $\Delta$LL")
    axes[2].set_ylabel(r"Held-out $\Delta$LL")
    fig.suptitle("Segment-level keylogging outcomes: predictor gain over controls", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "figure_keylogging_delta_llh_main.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "figure_keylogging_delta_llh_main.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_grouped_delta(table: pd.DataFrame, output_dir: Path) -> None:
    grouped = table[table["outcome"].isin(MAIN_OUTCOMES)].copy()
    if grouped.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    vmax = max(abs(grouped["metric"].min()), abs(grouped["metric"].max()), 0.005)
    for ax, outcome in zip(axes, MAIN_OUTCOMES):
        sub = grouped[grouped["outcome"] == outcome].copy()
        sub["feature"] = pd.Categorical(sub["feature"], PREDICTORS, ordered=True)
        sub["group"] = pd.Categorical(sub["group"], list(GROUPS.keys()), ordered=True)
        sub = sub.sort_values(["group", "feature"])
        pivot = sub.pivot(index="group", columns="feature", values="metric").reindex(index=list(GROUPS.keys()), columns=PREDICTORS)
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="vlag",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            cbar=ax is axes[-1],
            linewidths=0.4,
            linecolor="#f3f4f6",
        )
        ax.set_title(OUTCOME_LABELS[outcome], fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_yticklabels([GROUP_LABELS.get(g.get_text(), g.get_text()) for g in ax.get_yticklabels()], rotation=0, fontsize=8)
        ax.set_xticklabels([FEATURE_LABELS.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()], rotation=35, ha="right", fontsize=9)
        for _, row in sub[sub["sig_bh"]].iterrows():
            y = list(GROUPS.keys()).index(row["group"]) + 0.5
            x = PREDICTORS.index(row["feature"]) + 0.5
            ax.text(x, y, "*", ha="center", va="center", color="black", fontsize=11, fontweight="bold")
    fig.suptitle("Segment-level keylogging outcomes by task and direction", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "figure_keylogging_grouped.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "figure_keylogging_grouped.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def write_outputs(df: pd.DataFrame, output_dir: Path, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    main_outcomes = MAIN_OUTCOMES + ROBUSTNESS_OUTCOMES
    df.to_csv(output_dir / "keylogging_segment_dataset.tsv", sep="\t", index=False)
    preprocessing_summary(df, main_outcomes).to_csv(output_dir / "table_keylogging_preprocessing_summary.tsv", sep="\t", index=False)

    main_table = run_cv(df, "all", main_outcomes, n_splits=10, seed=seed)
    main_table.to_csv(output_dir / "figure_keylogging_delta_llh_main.tsv", sep="\t", index=False)
    plot_main_delta(main_table, output_dir)

    group_tables = []
    for group_name in GROUPS:
        group_tables.append(run_cv(df, group_name, main_outcomes, n_splits=3, seed=seed))
    grouped_table = pd.concat([t for t in group_tables if not t.empty], ignore_index=True)
    if not grouped_table.empty:
        grouped_table["q_bh"] = benjamini_hochberg(grouped_table["p_raw"].fillna(1.0).tolist())
        grouped_table["sig_bh"] = grouped_table["q_bh"] < 0.05
    grouped_table.to_csv(output_dir / "figure_keylogging_grouped.tsv", sep="\t", index=False)
    plot_grouped_delta(grouped_table, output_dir)

    robustness = main_table[main_table["outcome"].isin(ROBUSTNESS_OUTCOMES)].copy()
    robustness.to_csv(output_dir / "table_keylogging_robustness_nedit.tsv", sep="\t", index=False)

    sanity = run_cv(df, "all", SANITY_OUTCOMES, n_splits=10, seed=seed)
    sanity.to_csv(output_dir / "table_keylogging_logdur_sanity.tsv", sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run segment-level keylogging extension.")
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df, coverage = build_dataset(args.feature_dir, args.table_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.output_dir / "table_keylogging_merge_coverage.tsv", sep="\t", index=False)
    write_outputs(df, args.output_dir, args.seed)

    print(f"Merged rows: {len(df)}")
    print(f"Merged sessions: {df['session_id'].nunique()}")
    print(f"Merged translators: {df['translator_id'].nunique()}")
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
