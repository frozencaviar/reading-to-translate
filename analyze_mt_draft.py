#!/usr/bin/env python3
"""Compare MT draft features for edited vs unedited segments in PE conditions.

Tests: For each PE condition (P1, P2) and each MT-draft feature,
does the feature differ between segments that were edited (Dur > 0 for at
least one token) vs segments that were never edited?

Uses segment-level aggregation (mean per segment per participant), then
compares group means via two-sided permutation test.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MT_DRAFT_FEATURES = [
    "mt_draft_tgt_surp",
    "mt_draft_xent",
    "mt_draft_xmi",
    "mt_draft_eos_xatt",
    "mt_draft_y2srcseq_ent",
    "mt_draft_y2y",
    "mt_draft_y2cont",
    "mt_draft_y_ent",
    "mt_draft_x2cont",
    "mt_draft_cont2x",
    "mt_draft_x_ent",
    "mt_draft_x2x",
    "mt_draft_x2eos",
    "mt_draft_tgtseq2x",
]


def infer_condition(session_id: str) -> str | None:
    parts = str(session_id).split("_")
    if len(parts) < 2:
        return None
    cond = parts[1]
    if cond in ("P1", "P2"):
        return cond
    return None


def permutation_test(
    group_a: np.ndarray, group_b: np.ndarray, n_perm: int = 10000, seed: int = 42
) -> dict:
    """Two-sided permutation test for difference in means."""
    rng = np.random.default_rng(seed)
    obs_diff = np.mean(group_a) - np.mean(group_b)
    pooled = np.concatenate([group_a, group_b])
    n_a = len(group_a)

    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(pooled)
        perm_diffs[i] = np.mean(pooled[:n_a]) - np.mean(pooled[n_a:])

    p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
    return {
        "mean_edited": np.mean(group_a),
        "mean_unedited": np.mean(group_b),
        "diff": obs_diff,
        "p_value": p_value,
        "n_edited": n_a,
        "n_unedited": len(group_b),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent
        / "attention-norm-yiyu-positive-dur"
        / "yiyu-features-token-segment_with_mt_draft.tsv",
        help="TSV with MT draft features merged in",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "figures_yiyu_positive_dur_mixed"
        / "mt_draft_edit_prob_analysis.tsv",
        help="Output TSV for results",
    )
    parser.add_argument("--n-perm", type=int, default=10000, help="Permutation iterations")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input, sep="\t")
    print(f"Loaded: {len(df)} rows")

    # Filter to PE sessions only
    df["_cond"] = df["session_id"].apply(infer_condition)
    pe_df = df[df["_cond"].notna()].copy()
    print(f"PE rows: {len(pe_df)}")

    # Check which MT draft columns exist
    available_features = [f for f in MT_DRAFT_FEATURES if f in pe_df.columns]
    print(f"Available MT draft features: {available_features}")
    if not available_features:
        print("ERROR: No MT draft features found. Run merge_mt_draft_to_tsv.py first.")
        return

    # Segment-level aggregation: for each (session_id, tgt_segid), compute:
    # - edited: 1 if ANY token in this segment has Dur > 0, else 0
    # - mean of each MT draft feature (same across tokens in same segment, but take mean anyway)
    pe_df["_edited_token"] = (pe_df["Dur"] > 0).astype(int)
    seg_agg = (
        pe_df.groupby(["session_id", "_cond", "tgt_segid"])
        .agg(
            edited=("_edited_token", "max"),
            **{f: (f, "mean") for f in available_features},
        )
        .reset_index()
    )
    print(f"Segment-level aggregation: {len(seg_agg)} rows")

    # Run permutation tests per condition per feature
    results = []
    for cond in ["P1", "P2"]:
        cond_df = seg_agg[seg_agg["_cond"] == cond]
        edited = cond_df[cond_df["edited"] == 1]
        unedited = cond_df[cond_df["edited"] == 0]
        n_edited_segs = len(edited)
        n_unedited_segs = len(unedited)
        total_segs = len(cond_df)
        print(
            f"\n{cond}: {n_edited_segs} edited / {n_unedited_segs} unedited "
            f"(out of {total_segs} total segment-participant pairs)"
        )

        # Unique segment count
        unique_segs = cond_df["tgt_segid"].nunique()
        n_participants = cond_df["session_id"].nunique()
        print(f"  {unique_segs} unique segments × {n_participants} participants")

        for feat in available_features:
            edited_vals = edited[feat].dropna().values
            unedited_vals = unedited[feat].dropna().values

            if len(edited_vals) < 5 or len(unedited_vals) < 5:
                continue

            res = permutation_test(edited_vals, unedited_vals, n_perm=args.n_perm)
            results.append(
                {
                    "condition": cond,
                    "feature": feat,
                    "mean_edited": res["mean_edited"],
                    "mean_unedited": res["mean_unedited"],
                    "diff": res["diff"],
                    "p_value": res["p_value"],
                    "n_edited": res["n_edited"],
                    "n_unedited": res["n_unedited"],
                }
            )

    # Apply BH-FDR correction
    if results:
        res_df = pd.DataFrame(results)
        p_vals = res_df["p_value"].values
        n_tests = len(p_vals)
        ranked = np.argsort(p_vals)
        q_vals = np.empty(n_tests)
        for i, idx in enumerate(ranked):
            q_vals[idx] = min(p_vals[idx] * n_tests / (i + 1), 1.0)
        # Ensure monotonicity
        for i in range(n_tests - 2, -1, -1):
            q_vals[ranked[i]] = min(q_vals[ranked[i]], q_vals[ranked[i + 1]])
        res_df["q_bh"] = q_vals
        res_df["sig_bh"] = res_df["q_bh"] < 0.05

        # Sort by condition then p_value
        res_df = res_df.sort_values(["condition", "p_value"])

        # Print results
        print("\n=== RESULTS ===")
        for _, row in res_df.iterrows():
            sig = "***" if row["sig_bh"] else "   "
            print(
                f"{sig} {row['condition']} {row['feature']}: "
                f"edited={row['mean_edited']:.4f}, unedited={row['mean_unedited']:.4f}, "
                f"diff={row['diff']:+.4f}, p={row['p_value']:.4f}, q={row['q_bh']:.4f}"
            )

        # Summary
        n_sig = res_df["sig_bh"].sum()
        print(f"\nSignificant (q_bh < 0.05): {n_sig} / {len(res_df)}")

        # Write output
        args.output.parent.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(args.output, sep="\t", index=False)
        print(f"Written: {args.output}")
    else:
        print("No results to report.")


if __name__ == "__main__":
    main()
