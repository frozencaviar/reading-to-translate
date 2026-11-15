#!/usr/bin/env python3
"""Merge MT draft features into the existing Yiyu feature TSV files.

Reads mt_draft_features.tsv (segment-level MT draft features) and joins them
into the existing token-segment and segment-token TSVs by condition + tgt_segid.

New columns added: mt_draft_tgt_surp, mt_draft_xent, mt_draft_eos_xatt, etc.
"""

import argparse
from pathlib import Path

import pandas as pd


def infer_condition(session_id: str) -> str | None:
    """Map session_id to PE condition (P1 or P2)."""
    parts = str(session_id).split("_")
    if len(parts) < 2:
        return None
    cond = parts[1]
    if cond in ("P1", "P2"):
        return cond
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mt-draft",
        type=Path,
        default=Path(__file__).resolve().parent / "attention-norm-yiyu-positive-dur" / "mt_draft_features.tsv",
        help="MT draft features TSV",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "attention-norm-yiyu-positive-dur",
        help="Directory with yiyu-features-*.tsv files",
    )
    parser.add_argument(
        "--output-suffix",
        default="with_mt_draft",
        help="Suffix for output files (before .tsv)",
    )
    args = parser.parse_args()

    # Load MT draft features
    mt_df = pd.read_csv(args.mt_draft, sep="\t")
    print(f"Loaded MT draft features: {len(mt_df)} rows, columns: {list(mt_df.columns)}")

    # Identify MT draft columns (all except condition, tgt_segid)
    mt_cols = [c for c in mt_df.columns if c not in ("condition", "tgt_segid")]
    print(f"MT draft feature columns: {mt_cols}")

    # Process each TSV file
    for fname in ["yiyu-features-token-segment.tsv", "yiyu-features-segment-token.tsv"]:
        tsv_path = args.data_dir / fname
        if not tsv_path.exists():
            print(f"  SKIP: {fname} not found")
            continue

        df = pd.read_csv(tsv_path, sep="\t")
        print(f"\nProcessing {fname}: {len(df)} rows")

        # Map session_id to PE condition
        df["_condition"] = df["session_id"].apply(infer_condition)
        pe_mask = df["_condition"].notna()
        pe_count = pe_mask.sum()
        print(f"  PE rows: {pe_count} / {len(df)}")

        if pe_count == 0:
            df.drop(columns=["_condition"], inplace=True)
            continue

        # Ensure tgt_segid types match for join
        # MT draft uses 1-indexed seg IDs; data uses 0-indexed
        df["_tgt_segid"] = df["tgt_segid"].astype(int).astype(str)
        mt_df_clean = mt_df.copy()
        mt_df_clean["_tgt_segid"] = (mt_df_clean["tgt_segid"].astype(int) - 1).astype(str)

        # Join MT draft features
        for mt_col in mt_cols:
            # Build a lookup: (condition, tgt_segid) -> value
            lookup = mt_df_clean.set_index(["condition", "_tgt_segid"])[mt_col].to_dict()
            df.loc[pe_mask, mt_col] = df.loc[pe_mask].apply(
                lambda row: lookup.get((row["_condition"], row["_tgt_segid"]), float("nan")),
                axis=1,
            )
            filled = pe_mask & df[mt_col].notna()
            print(f"  {mt_col}: filled {filled.sum()} / {pe_count} PE rows")

        # Clean up temporary columns
        df.drop(columns=["_condition", "_tgt_segid"], inplace=True)

        # Write output
        out_path = tsv_path.with_name(fname.replace(".tsv", f"_{args.output_suffix}.tsv"))
        df.to_csv(out_path, sep="\t", index=False, na_rep="nan")
        print(f"  Written: {out_path}")


if __name__ == "__main__":
    main()
