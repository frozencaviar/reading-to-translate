#!/usr/bin/env python3
"""Run a two-part mixed model for target production duration.

Part 1 uses a logistic GLMM for whether a unit was edited at all.
Part 2 uses a Gaussian mixed model for log duration conditional on positive
duration. The fitting itself is delegated to R/lme4.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore

try:
    from wordfreq import word_frequency

    FREQ_SOURCE = "wordfreq"
except Exception:  # pragma: no cover
    word_frequency = None
    FREQ_SOURCE = "corpus_unigram_fallback"


FEATURES = ["tgt_surp", "xent", "eos_xatt", "y2srcseq_ent", "y2y", "y2cont", "y_ent"]
CONTROLS = ["tgt_seg_len", "tgt_freq", "tgt_posq"]
MAIN_GROUPS = ["all", "all_translating", "all_post-editing", "all_en-zh", "all_zh-en"]
SPLIT_GROUPS = ["translating_en-zh", "translating_zh-en", "post-editing_en-zh", "post-editing_zh-en"]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=script_dir / "attention-norm-yiyu-positive-dur")
    parser.add_argument("--outdir", type=Path, default=script_dir / "figures_yiyu_positive_dur_twopart_glmm")
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--positive-min-ms", type=float, default=20.0)
    parser.add_argument("--maxfun", type=int, default=100000)
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
        for tokens, lang in frame[["tgt_tokens", "tgt_lang"]].dropna().itertuples(index=False):
            lang = str(lang)
            counts.setdefault(lang, {})
            for token in str(tokens).split("_"):
                if not token or token == "---":
                    continue
                counts[lang][token] = counts[lang].get(token, 0) + 1
                totals[lang] = totals.get(lang, 0) + 1
    return {
        lang: {token: count / max(totals.get(lang, 0), 1) for token, count in lang_counts.items()}
        for lang, lang_counts in counts.items()
    }


def mean_log_frequency(tokens: object, lang: object, fallback: dict[str, dict[str, float]]) -> float:
    pieces = [w for w in str(tokens).split("_") if w and w != "---"]
    if not pieces:
        return np.nan
    lang = str(lang)
    if word_frequency is not None:
        vals = [max(word_frequency(piece, lang), 1e-8) for piece in pieces]
    else:
        vals = [max(fallback.get(lang, {}).get(piece, 1e-8), 1e-8) for piece in pieces]
    return float(np.log(np.mean(vals)))


def add_common_columns(df: pd.DataFrame, fallback_freqs: dict[str, dict[str, float]], level: str, positive_min_ms: float) -> pd.DataFrame:
    out = df.copy()
    out["level"] = level
    out["Dur"] = pd.to_numeric(out["Dur"], errors="coerce")
    out["edited"] = (out["Dur"] > 0).astype(int)
    positive_mask = (out["Dur"] > 0) & (out["Dur"] >= positive_min_ms)
    out["log_Dur_positive"] = np.nan
    out.loc[positive_mask, "log_Dur_positive"] = np.log(out.loc[positive_mask, "Dur"])
    out["log_Dur_positive"] = zscore_col(out["log_Dur_positive"])
    out["lang_pair"] = out[["src_lang", "tgt_lang"]].apply(lambda x: "-".join(x.astype(str)), axis=1)
    out["task_type"] = out["session_id"].apply(infer_task_type)
    out["task_code"] = out["session_id"].apply(infer_task_code)
    out["translator_id"] = out["session_id"].apply(lambda s: str(s).split("_")[0])
    out["material_id"] = out[["study", "task_code"]].apply(lambda x: f"{x.iloc[0]}::{x.iloc[1]}", axis=1)
    out["sentence_item_id"] = out[["material_id", "src_segid"]].apply(lambda x: f"{x.iloc[0]}::s{x.iloc[1]}", axis=1)
    out["tgt_seg_len"] = out["tgt_token_ids"].apply(lambda ids: len(str(ids).split("+")) if pd.notnull(ids) else np.nan)
    out["tgt_freq"] = out[["tgt_tokens", "tgt_lang"]].apply(
        lambda x: mean_log_frequency(x.iloc[0], x.iloc[1], fallback_freqs),
        axis=1,
    )
    if "y_ent" in out.columns:
        out["y_ent"] = out["y_ent"].fillna(0)
    for col in CONTROLS + FEATURES:
        if col in out.columns:
            out[col] = zscore_col(out[col])
    keep = [
        "level",
        "study",
        "session_id",
        "translator_id",
        "sentence_item_id",
        "task_type",
        "task_code",
        "lang_pair",
        "Dur",
        "edited",
        "log_Dur_positive",
    ] + CONTROLS + FEATURES
    return out[keep].copy()


def prepare_model_data(data_dir: Path, positive_min_ms: float) -> pd.DataFrame:
    tok_seg = pd.read_csv(data_dir / "yiyu-features-token-segment.tsv", sep="\t")
    seg_tok = pd.read_csv(data_dir / "yiyu-features-segment-token.tsv", sep="\t")
    fallback_freqs = corpus_frequency_maps([tok_seg, seg_tok])
    token_df = add_common_columns(seg_tok, fallback_freqs, "token", positive_min_ms)
    segment_raw = tok_seg.drop_duplicates(["study", "session_id", "src_segid", "tgt_token_ids"]).copy()
    segment_df = add_common_columns(segment_raw, fallback_freqs, "segment", positive_min_ms)
    return pd.concat([token_df, segment_df], ignore_index=True)


def write_plot_tables(all_tests: pd.DataFrame, outdir: Path) -> None:
    base_cols = ["group", "side", "label", "level", "predictor", "feature", "metric", "p_raw", "n_participants", "q_bh", "sig_bh"]
    for part in ["edit_probability", "positive_duration"]:
        part_dir = outdir / part
        part_dir.mkdir(parents=True, exist_ok=True)
        part_df = all_tests[(all_tests["outcome_part"] == part) & (all_tests["status"] == "ok")].copy()
        # Compatibility shim for plot_delta_llh_grouped_topjournal.py: the
        # duration panel is keyed by side=tgt and label=log_Dur.
        part_df["label"] = "log_Dur"
        main = part_df[part_df["group"].isin(MAIN_GROUPS)][base_cols].copy()
        split = part_df[part_df["group"].isin(SPLIT_GROUPS)][base_cols].copy()
        fig1 = part_df[part_df["group"].eq("all")][base_cols[1:]].copy()
        main.to_csv(part_dir / "figure5_delta_llh_grouped.tsv", sep="\t", index=False)
        split.to_csv(part_dir / "figure5_appendix_delta_llh_grouped_split.tsv", sep="\t", index=False)
        fig1.to_csv(part_dir / "figure1_delta_llh_main.tsv", sep="\t", index=False)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    prepared_path = args.outdir / "table_twopart_prepared_model_data.tsv"
    all_tests_path = args.outdir / "table_twopart_glmm_all_tests.tsv"
    metadata_path = args.outdir / "table_twopart_glmm_metadata.tsv"
    r_model_script = Path(__file__).resolve().with_name("run_two_part_glmm_dur_models.R")

    prepared = prepare_model_data(args.data_dir, args.positive_min_ms)
    prepared.to_csv(prepared_path, sep="\t", index=False)

    cmd = [args.rscript, str(r_model_script), str(prepared_path), str(all_tests_path), str(args.maxfun)]
    subprocess.run(cmd, check=True)

    all_tests = pd.read_csv(all_tests_path, sep="\t")
    write_plot_tables(all_tests, args.outdir)
    metadata = pd.DataFrame([
        {"key": "part1", "value": "edited ~ controls + predictor + task/direction if varying, binomial GLMM; correlated maximal by-participant and by-item random slopes for the focal predictor (Barr et al., 2013), reduced by removing correlations (||), then dropping the by-item slope, then to random intercepts on nonconvergence/singularity"},
        {"key": "part2", "value": "log_Dur_positive ~ controls + predictor + task/direction if varying, ML Gaussian LMM; correlated maximal by-participant and by-item random slopes for the focal predictor (Barr et al., 2013), reduced by removing correlations (||), then dropping the by-item slope, then to random intercepts on nonconvergence/singularity"},
        {"key": "edited_definition", "value": "Dur > 0"},
        {"key": "positive_duration_filter", "value": f"Dur > 0 and Dur >= {args.positive_min_ms:g} ms"},
        {"key": "metric", "value": "(logLik(control+predictor) - logLik(control)) / nobs"},
        {"key": "p_raw", "value": "likelihood-ratio chi-square test"},
        {"key": "frequency_source", "value": FREQ_SOURCE},
    ])
    metadata.to_csv(metadata_path, sep="\t", index=False)
    print(f"[two-part] wrote outputs to {args.outdir}")


if __name__ == "__main__":
    main()
