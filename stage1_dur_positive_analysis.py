"""
Exploratory Stage-1 model for target duration:
  1) Binary outcome: whether production duration is positive and >= 20ms (Dur>0)
  2) Pooled logistic GLM with controls + predictor
  3) Pooled interaction logistic GLM (predictor x task / predictor x direction)

Outputs are written to figures_yiyu_positive_dur as standalone exploratory artifacts.
"""

import math
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import statsmodels.api as sm
import statsmodels.formula.api as smf

import reproduce_yiyu_analysis as rya


OUT_DIR = rya.FIGURES_DIR
EPS = 1e-10
TARGET_SIDE = "tgt"
LEVELS = ["token", "segment"]
INTERACTION_TYPES = ["task", "direction"]


def bernoulli_loglik(prob, y):
    p = np.clip(np.asarray(prob, dtype=float), EPS, 1.0 - EPS)
    yy = np.asarray(y, dtype=float)
    return yy * np.log(p) + (1.0 - yy) * np.log(1.0 - p)


def fit_binomial_formula_model(train, test, formula, label, required_cols):
    train = train.copy()
    test = test.copy()
    if len(train) < 30 or len(test) < 10:
        return None

    train = train.dropna(subset=[c for c in required_cols + [label] if c in train.columns])
    test = test.dropna(subset=[c for c in required_cols + [label] if c in test.columns])
    if len(train) < 30 or len(test) < 10:
        return None

    # Keep outcome as numeric 0/1
    train[label] = pd.to_numeric(train[label], errors="coerce")
    test[label] = pd.to_numeric(test[label], errors="coerce")
    train = train.dropna(subset=[label])
    test = test.dropna(subset=[label])
    if len(train) < 30 or len(test) < 10:
        return None

    try:
        model = smf.glm(
            formula=formula,
            data=train,
            family=sm.families.Binomial(),
        )
        fit = model.fit()
    except Exception:
        return None

    try:
        probs = np.asarray(fit.predict(test), dtype=float)
    except Exception:
        return None

    llh = bernoulli_loglik(probs, test[label].values).tolist()
    meta_cols = ["row_uid", "translator_id", "sentence_id", "session_id", "task_type", "lang_pair"]
    meta_cols = [c for c in meta_cols if c in test.columns]
    meta = test[meta_cols].copy()
    meta["llh"] = llh
    meta["prob"] = probs
    meta["y"] = test[label].values
    return {"fit": fit, "test_llh": llh, "test_meta": meta}


def add_stage1_label(df):
    out = df.copy()
    if "valid_Dur_positive_ge20" in out.columns:
        out["dur_positive"] = out["valid_Dur_positive_ge20"].astype(int)
    else:
        out["dur_positive"] = ((out["Dur"] > 0) & (out["Dur"] >= rya.DUR_MIN_MS)).astype(int)
    return out


def main_effect_delta(train, test, ctrl_feats, predictor):
    label = "dur_positive"
    add_formula = f"{label} ~ " + " + ".join(ctrl_feats + ["C(task_type)", "C(lang_pair)"])
    pred_formula = f"{label} ~ " + " + ".join(ctrl_feats + [predictor, "C(task_type)", "C(lang_pair)"])
    required = ctrl_feats + [predictor, "task_type", "lang_pair"]

    add_res = fit_binomial_formula_model(train, test, add_formula, label, required)
    if add_res is None:
        return None
    pred_res = fit_binomial_formula_model(train, test, pred_formula, label, required)
    if pred_res is None:
        return None

    add_meta = add_res["test_meta"]
    pred_meta = pred_res["test_meta"]
    if "row_uid" not in add_meta.columns or "row_uid" not in pred_meta.columns:
        delta = [p - b for p, b in zip(pred_res["test_llh"], add_res["test_llh"])]
        return {"test_llh_delta_df": pd.DataFrame({"delta": delta})}

    merged = pred_meta[["row_uid", "llh"]].rename(columns={"llh": "llh_pred"}).merge(
        add_meta[["row_uid", "llh"]].rename(columns={"llh": "llh_add"}),
        on="row_uid",
        how="inner",
    )
    merged["delta"] = merged["llh_pred"] - merged["llh_add"]
    meta_cols = ["row_uid"] + [c for c in ["translator_id", "sentence_id", "session_id", "task_type", "lang_pair"] if c in pred_meta.columns]
    out = pred_meta[meta_cols].merge(merged[["row_uid", "delta"]], on="row_uid", how="inner")
    return {"test_llh_delta_df": out}


def interaction_delta(train, test, ctrl_feats, predictor, interaction="task"):
    label = "dur_positive"
    additive_formula = f"{label} ~ " + " + ".join(ctrl_feats + [predictor, "C(task_type)", "C(lang_pair)"])
    if interaction == "task":
        int_formula = f"{label} ~ " + " + ".join(ctrl_feats + [f"{predictor} * C(task_type)", "C(lang_pair)"])
    elif interaction == "direction":
        int_formula = f"{label} ~ " + " + ".join(ctrl_feats + [f"{predictor} * C(lang_pair)", "C(task_type)"])
    else:
        raise ValueError(interaction)

    required = ctrl_feats + [predictor, "task_type", "lang_pair"]
    add_res = fit_binomial_formula_model(train, test, additive_formula, label, required)
    if add_res is None:
        return None
    int_res = fit_binomial_formula_model(train, test, int_formula, label, required)
    if int_res is None:
        return None

    add_meta = add_res["test_meta"]
    int_meta = int_res["test_meta"]
    if "row_uid" not in add_meta.columns or "row_uid" not in int_meta.columns:
        delta = [i - a for i, a in zip(int_res["test_llh"], add_res["test_llh"])]
        return {"test_llh_delta_df": pd.DataFrame({"delta": delta})}

    merged = int_meta[["row_uid", "llh"]].rename(columns={"llh": "llh_int"}).merge(
        add_meta[["row_uid", "llh"]].rename(columns={"llh": "llh_add"}),
        on="row_uid",
        how="inner",
    )
    merged["delta"] = merged["llh_int"] - merged["llh_add"]
    meta_cols = ["row_uid"] + [c for c in ["translator_id", "sentence_id", "session_id", "task_type", "lang_pair"] if c in int_meta.columns]
    out = int_meta[meta_cols].merge(merged[["row_uid", "delta"]], on="row_uid", how="inner")
    return {"test_llh_delta_df": out}


def participant_values(results_list):
    frames = []
    for res in results_list:
        if res is None:
            continue
        df = res.get("test_llh_delta_df")
        if isinstance(df, pd.DataFrame) and not df.empty and "translator_id" in df.columns and "delta" in df.columns:
            frames.append(df[["translator_id", "delta"]].copy())
    if not frames:
        return np.array([], dtype=float), 0
    joined = pd.concat(frames, ignore_index=True)
    per = joined.groupby("translator_id", as_index=False)["delta"].mean()
    vals = per["delta"].to_numpy(dtype=float)
    return vals, int(len(per))


def summarize_with_bh(raw_records):
    df = pd.DataFrame(raw_records)
    if df.empty:
        return df
    qvals = rya.benjamini_hochberg_adjust(df["p_raw"].to_numpy(dtype=float))
    df["q_bh"] = qvals
    df["sig_bh"] = df["q_bh"] < 0.05
    return df


def plot_stage1(df, out_base, title):
    if df.empty:
        return
    predictor_order = [ft for ft in rya.features[TARGET_SIDE] if ft in set(df["feature"])]
    x = np.arange(len(predictor_order))
    level_styles = {"token": ("o", -0.10), "segment": ("^", 0.10)}

    fig, ax = plt.subplots(1, 1, figsize=(9.8, 3.8))
    for level, (marker, off) in level_styles.items():
        sub = df[df["level"] == level].set_index("feature").reindex(predictor_order).reset_index()
        xs = x + off
        ys = sub["metric"].to_numpy(dtype=float)
        sig = sub["sig_bh"].fillna(False).to_numpy()
        for xi, yi, s in zip(xs, ys, sig):
            ax.scatter(
                xi, yi, marker=marker, s=48,
                facecolors=("#353535" if s else "none"),
                edgecolors="#353535", linewidths=1.2, zorder=20
            )

    ax.axhline(0, ls="--", color="#d98b2b", lw=1.0)
    ax.grid(axis="y", color="0.93", linewidth=0.8)
    ax.set_xlim(-0.6, len(predictor_order) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([rya.feature_names.get(ft, ft) for ft in predictor_order], rotation=20, ha="right", fontsize=11)
    ax.set_ylabel(r"$\Delta$ LL", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis="y", labelsize=10.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.35")
        spine.set_linewidth(0.9)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Token", markerfacecolor="#353535", markeredgecolor="#353535", linestyle="None", markersize=6),
        Line2D([0], [0], marker="^", color="w", label="Segment", markerfacecolor="#353535", markeredgecolor="#353535", linestyle="None", markersize=6),
        Line2D([0], [0], marker="o", color="w", label="Significant", markerfacecolor="#353535", markeredgecolor="#353535", linestyle="None", markersize=6),
        Line2D([0], [0], marker="o", color="w", label="Not significant", markerfacecolor="white", markeredgecolor="#353535", linestyle="None", markersize=6),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, frameon=True, facecolor="white", edgecolor="0.85")

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{out_base}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}/{out_base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_stage1_interaction(df):
    if df.empty:
        return
    predictor_order = [ft for ft in rya.features[TARGET_SIDE] if ft in set(df["feature"])]
    x = np.arange(len(predictor_order))
    level_styles = {"token": ("o", -0.10), "segment": ("^", 0.10)}
    interactions = ["task", "direction"]
    titles = {
        "task": "Stage-1 Dur>0: predictor × task gain",
        "direction": "Stage-1 Dur>0: predictor × direction gain",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.0), sharey=True)
    all_vals = df["metric"].to_numpy(dtype=float)
    ymin, ymax = rya.compute_shared_ylim(all_vals, min_span=0.03)

    for ax, interaction in zip(axes, interactions):
        subi = df[df["interaction_type"] == interaction].copy()
        for level, (marker, off) in level_styles.items():
            sub = subi[subi["level"] == level].set_index("feature").reindex(predictor_order).reset_index()
            xs = x + off
            ys = sub["metric"].to_numpy(dtype=float)
            sig = sub["sig_bh"].fillna(False).to_numpy()
            for xi, yi, s in zip(xs, ys, sig):
                ax.scatter(
                    xi, yi, marker=marker, s=46,
                    facecolors=("#3b3b3b" if s else "none"),
                    edgecolors="#3b3b3b", linewidths=1.2, zorder=20
                )
        ax.axhline(0, ls="--", color="#d98b2b", lw=1.0)
        ax.grid(axis="y", color="0.93", linewidth=0.8)
        ax.set_xlim(-0.6, len(predictor_order) - 0.4)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels([rya.feature_names.get(ft, ft) for ft in predictor_order], rotation=20, ha="right", fontsize=10.5)
        ax.set_title(titles[interaction], fontsize=12)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("0.35")
            spine.set_linewidth(0.9)
    axes[0].set_ylabel(r"$\Delta LL_{\mathrm{int}}$", fontsize=12)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Token", markerfacecolor="#3b3b3b", markeredgecolor="#3b3b3b", linestyle="None", markersize=6),
        Line2D([0], [0], marker="^", color="w", label="Segment", markerfacecolor="#3b3b3b", markeredgecolor="#3b3b3b", linestyle="None", markersize=6),
        Line2D([0], [0], marker="o", color="w", label="Significant", markerfacecolor="#3b3b3b", markeredgecolor="#3b3b3b", linestyle="None", markersize=6),
        Line2D([0], [0], marker="o", color="w", label="Not significant", markerfacecolor="white", markeredgecolor="#3b3b3b", linestyle="None", markersize=6),
    ]
    fig.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=4, fontsize=10, frameon=True)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(f"{OUT_DIR}/dur-positive-stage1-interaction.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}/dur-positive-stage1-interaction.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_stage1():
    print("Loading data for Stage-1 Dur>0 analysis...")
    data = rya.get_data(rya.DATA_DIR, standardize=True)
    trains, tests = rya.get_train_test_split(data, n_splits=rya.MAIN_N_SPLITS, random_seed=42)
    if not trains:
        raise RuntimeError("No train/test splits available.")

    tgt_predictors = list(rya.features[TARGET_SIDE])
    ctrl = rya.control_features[TARGET_SIDE]

    main_store = {level: defaultdict(list) for level in LEVELS}
    int_store = {itype: {level: defaultdict(list) for level in LEVELS} for itype in INTERACTION_TYPES}

    print("Running CV folds for Stage-1 logistic models...")
    for fold_idx, (tr, te) in enumerate(zip(trains, tests), start=1):
        print(f"  Fold {fold_idx}/{len(trains)}")
        for level in LEVELS:
            train_df = add_stage1_label(tr[TARGET_SIDE][level])
            test_df = add_stage1_label(te[TARGET_SIDE][level])
            for predictor in tgt_predictors:
                res_main = main_effect_delta(train_df, test_df, ctrl[level], predictor)
                if res_main is not None:
                    main_store[level][predictor].append(res_main)
                for itype in INTERACTION_TYPES:
                    res_int = interaction_delta(train_df, test_df, ctrl[level], predictor, interaction=itype)
                    if res_int is not None:
                        int_store[itype][level][predictor].append(res_int)

    print("Aggregating participant-level statistics...")
    main_records = []
    for level in LEVELS:
        for predictor in tgt_predictors:
            vals, n_parts = participant_values(main_store[level][predictor])
            if vals.size == 0:
                continue
            p_raw = rya.signflip_permutation_pvalue(vals, n_resamples=rya.PERMUTATION_N, rng_obj=rya.rng)
            main_records.append({
                "level": level,
                "predictor": f"control+{predictor}",
                "feature": predictor,
                "metric": float(np.mean(vals)),
                "p_raw": float(p_raw),
                "n_participants": int(n_parts),
            })
    main_df = summarize_with_bh(main_records)

    int_records = []
    for itype in INTERACTION_TYPES:
        for level in LEVELS:
            for predictor in tgt_predictors:
                vals, n_parts = participant_values(int_store[itype][level][predictor])
                if vals.size == 0:
                    continue
                p_raw = rya.signflip_permutation_pvalue(vals, n_resamples=rya.PERMUTATION_N, rng_obj=rya.rng)
                int_records.append({
                    "interaction_type": itype,
                    "level": level,
                    "predictor": f"control+{predictor}",
                    "feature": predictor,
                    "metric": float(np.mean(vals)),
                    "p_raw": float(p_raw),
                    "n_participants": int(n_parts),
                })
    int_df = summarize_with_bh(int_records)

    main_tsv = f"{OUT_DIR}/table_stage1_dur_positive_main.tsv"
    int_tsv = f"{OUT_DIR}/figure_stage1_dur_positive_interaction.tsv"
    main_df.to_csv(main_tsv, sep="\t", index=False)
    int_df.to_csv(int_tsv, sep="\t", index=False)
    print(f"Saved {main_tsv}")
    print(f"Saved {int_tsv}")

    print("Rendering Stage-1 figures...")
    plot_stage1(main_df, "dur-positive-stage1-main", "Stage-1 Dur>0: predictor gain vs control")
    plot_stage1_interaction(int_df)
    print(f"Saved {OUT_DIR}/dur-positive-stage1-main.png")
    print(f"Saved {OUT_DIR}/dur-positive-stage1-interaction.png")

    # Compact console summary for quick check
    if not main_df.empty:
        print("\nTop Stage-1 main gains:")
        print(
            main_df.sort_values("metric", ascending=False)[
                ["level", "feature", "metric", "q_bh", "sig_bh", "n_participants"]
            ].head(8).to_string(index=False)
        )
    if not int_df.empty:
        print("\nTop Stage-1 interaction gains:")
        print(
            int_df.sort_values("metric", ascending=False)[
                ["interaction_type", "level", "feature", "metric", "q_bh", "sig_bh", "n_participants"]
            ].head(10).to_string(index=False)
        )


if __name__ == "__main__":
    run_stage1()
