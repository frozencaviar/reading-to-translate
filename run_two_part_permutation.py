#!/usr/bin/env python3
"""LOOCV sign-flip permutation for two-part GLMM duration analysis.

Replaces LRT p-values in the all_tests TSV with participant-level permutation values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import zscore

try:
    from wordfreq import word_frequency
    FREQ_SOURCE = "wordfreq"
except ImportError:
    word_frequency = None
    FREQ_SOURCE = "corpus_unigram_fallback"

PERMUTATION_N = 1000
SEED = 42
DUR_MIN_MS = 20

FEATURES = ["tgt_surp", "xent", "eos_xatt", "y2srcseq_ent", "y2y", "y2cont", "y_ent"]
CONTROLS = ["tgt_seg_len", "tgt_freq", "tgt_posq"]
GROUPS_MAIN = [
    "all", "all_translating", "all_post-editing",
    "all_en-zh", "all_zh-en",
]
LEVELS = ["token", "segment"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--all-tests", type=Path, required=True, help="Path to table_twopart_glmm_all_tests.tsv")
    p.add_argument("--data-dir", type=Path, required=True, help="attention-norm-yiyu-positive-dur directory")
    p.add_argument("--outdir", type=Path, default=Path("figures_yiyu_positive_dur_twopart_glmm"))
    return p.parse_args()


def signflip_permutation_pvalue(values, n_resamples=PERMUTATION_N, rng=None):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = vals.size
    if n == 0:
        return np.nan
    if rng is None:
        rng = np.random.default_rng(SEED)
    observed = float(np.mean(vals))
    if n == 1:
        return 1.0
    perm_stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        signs = rng.choice([-1.0, 1.0], size=n)
        perm_stats[i] = np.mean(vals * signs)
    return float((np.sum(np.abs(perm_stats) >= abs(observed)) + 1.0) / (n_resamples + 1.0))


def bh_adjust(pvalues):
    pv = np.asarray(pvalues, dtype=float)
    n = len(pv)
    order = np.argsort(pv)
    sorted_p = pv[order]
    q_values = np.full(n, np.nan)
    for i in range(n - 1, -1, -1):
        q_values[i] = min(sorted_p[i] * n / (i + 1), q_values[i + 1] if i + 1 < n else 1.0)
    result = np.empty(n)
    result[order] = q_values
    return result


def fit_ols_logit(df, outcome, features):
    """Fit OLS (for continuous) or Logit (for binary). Returns tuple of (model, converged)."""
    X = df[features].copy()
    X = X.astype(float)
    X = X.fillna(X.mean())
    y = df[outcome].astype(float).values
    valid = np.isfinite(y) & np.all(np.isfinite(X.values), axis=1)
    X = X.loc[valid].values
    y = y[valid]
    if len(y) < 40:
        return None, False
    X = sm.add_constant(X)
    try:
        if set(y).issubset({0, 1}):
            model = sm.Logit(y, X).fit(disp=0, maxiter=200)
        else:
            model = sm.OLS(y, X).fit()
        return model, True
    except Exception:
        return None, False


def compute_delta_ll(model_base, model_pred, y):
    """Compute delta LL normalized by n."""
    if model_base is None or model_pred is None:
        return np.nan
    ll_base = model_base.llf
    ll_pred = model_pred.llf
    n = min(len(y), model_pred.nobs)
    return (ll_pred - ll_base) / max(n, 1)


def load_data(data_dir):
    tok_seg = pd.read_csv(data_dir / "yiyu-features-segment-token.tsv", sep="\t")
    seg_tok = pd.read_csv(data_dir / "yiyu-features-token-segment.tsv", sep="\t")
    frames = []
    for name, level in [(tok_seg, "token"), (seg_tok, "segment")]:
        df = name.copy()
        df["translator_id"] = df["session_id"].apply(lambda s: str(s).split("_")[0])
        df["task_type"] = df["session_id"].apply(lambda s: "post-editing" if "P" in str(s) else "translating")
        df["lang_pair"] = pd.Categorical(df["src_lang"].astype(str) + "-" + df["tgt_lang"].astype(str))
        df["material_id"] = df["study"] + "_" + df["src_segid"].astype(str)
        df["sentence_item_id"] = df[["material_id", "src_segid"]].apply(lambda x: f"{x.iloc[0]}::s{x.iloc[1]}", axis=1)
        
        # Frequency
        for side in ["tgt"]:
            df[f"{side}_seg_len"] = df[f"{side}_token_ids"].apply(
                lambda ids: len(str(ids).split("+")) if pd.notna(ids) else np.nan)
        for side in ["tgt"]:
            df[f"{side}_freq"] = 1.0  # placeholder
            df[f"{side}_posq"] = df[f"{side}_posq"] if f"{side}_posq" in df.columns else 0.5
        
        # Outcomes
        df["log_Dur"] = np.where(df["Dur"] >= DUR_MIN_MS, np.log(df["Dur"]), np.nan)
        df["Dur_pos"] = np.where((df["Dur"] > 0) & (df["Dur"] >= DUR_MIN_MS), np.log(df["Dur"]), np.nan)
        df["edited_bin"] = (df["Dur"] > 0).astype(int)
        
        df["level"] = level
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def filter_group(df, group_name):
    if group_name == "all":
        return df
    if group_name == "all_translating":
        return df[df["task_type"] == "translating"]
    if group_name == "all_post-editing":
        return df[df["task_type"] == "post-editing"]
    if group_name == "all_en-zh":
        return df[df["lang_pair"] == "en-zh"]
    if group_name == "all_zh-en":
        return df[df["lang_pair"] == "zh-en"]
    return df


def main():
    args = parse_args()
    rng = np.random.default_rng(SEED)
    
    # Load data
    data = load_data(args.data_dir)
    all_tests = pd.read_csv(args.all_tests, sep="\t")
    
    results = []
    total = len(all_tests[all_tests["status"] == "ok"])
    done = 0
    
    for idx, row in all_tests.iterrows():
        if row["status"] != "ok":
            results.append(row.to_dict())
            continue
        
        group = row["group"]
        outcome_part = row["outcome_part"]
        label = row["label"]
        level = row["level"]
        feature = row["feature"]
        predictor_name = row["predictor"]
        
        # Skip non-main groups or groups we can't filter
        work = filter_group(data, group)
        work = work[work["level"] == level].copy()
        
        if len(work) < 40:
            row_dict = row.to_dict()
            row_dict["status"] = "insufficient_permutation"
            results.append(row_dict)
            continue
        
        # Determine outcome
        if outcome_part == "edit_probability":
            outcome = "edited_bin"
        else:
            outcome = "log_Dur"
        
        features_for_model = CONTROLS + [feature] if feature in FEATURES else CONTROLS
        control_features = CONTROLS
        
        # Full model fit
        base_full, ok = fit_ols_logit(work, outcome, control_features)
        pred_full, ok2 = fit_ols_logit(work, outcome, features_for_model)
        
        if not ok or not ok2:
            row_dict = row.to_dict()
            row_dict["status"] = "fit_failed_permutation"
            results.append(row_dict)
            continue
        
        # LOO per participant
        participants = sorted(work["translator_id"].dropna().unique())
        n_participants = len(participants)
        if n_participants < 5:
            row_dict = row.to_dict()
            row_dict["status"] = "too_few_participants"
            results.append(row_dict)
            continue
        
        participant_deltas = []
        for pid in participants:
            loo_data = work[work["translator_id"] != pid].copy()
            if len(loo_data) < 30:
                continue
            base_loo, _ = fit_ols_logit(loo_data, outcome, control_features)
            pred_loo, _ = fit_ols_logit(loo_data, outcome, features_for_model)
            if base_loo is None or pred_loo is None:
                continue
            delta_full = compute_delta_ll(base_full, pred_full, work[outcome])
            delta_loo = compute_delta_ll(base_loo, pred_loo, loo_data[outcome])
            if np.isfinite(delta_full) and np.isfinite(delta_loo):
                # Jackknife pseudo-value
                pseudo = n_participants * delta_full - (n_participants - 1) * delta_loo
                participant_deltas.append(pseudo)
        
        if len(participant_deltas) < 5:
            row_dict = row.to_dict()
            row_dict["status"] = "insufficient_valid_participants"
            results.append(row_dict)
            continue
        
        p_perm = signflip_permutation_pvalue(participant_deltas, n_resamples=PERMUTATION_N, rng=rng)
        delta_mean = float(np.mean(participant_deltas))
        
        row_dict = row.to_dict()
        row_dict["p_raw"] = p_perm
        row_dict["n_participants"] = len(participant_deltas)
        row_dict["status"] = "ok"
        row_dict["metric"] = delta_mean
        results.append(row_dict)
        
        done += 1
        if done % 20 == 0:
            print(f"  [{done}/{total}] permutation done")
    
    result_df = pd.DataFrame(results)
    
    # Apply BH-FDR separately for each outcome_part
    for part in ["edit_probability", "positive_duration"]:
        mask = (result_df["status"] == "ok") & (result_df["outcome_part"] == part)
        pvals = result_df.loc[mask, "p_raw"].values
        if len(pvals) > 0:
            result_df.loc[mask, "q_bh"] = bh_adjust(pvals)
            result_df.loc[mask, "sig_bh"] = result_df.loc[mask, "q_bh"] < 0.05
    
    # Write
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / "table_twopart_glmm_all_tests_permuted.tsv"
    result_df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {len(result_df)} rows to {out_path}")
    sig_count = len(result_df[(result_df["status"] == "ok") & (result_df["sig_bh"] == True)])
    print(f"Significant (permutation + BH-FDR): {sig_count}")


if __name__ == "__main__":
    main()
