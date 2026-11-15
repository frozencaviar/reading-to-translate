#!/usr/bin/env python3
"""Random-slope sensitivity: refits 6 core models with by-participant random slopes.

Tiers (Barr et al., 2013):
  1) maximal — intercept + slope, vc_formula
  2) uncorrelated — separate vc terms
  3) RI-only — current baseline
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

MODELS = [
    ("src","log_TrtS","token",["src_seg_len","src_freq","src_posq"],"src_surp"),
    ("src","log_TrtS","segment",["src_seg_len","src_freq","src_posq"],"x2x"),
    ("tgt","log_TrtT","token",["tgt_seg_len","tgt_freq","tgt_posq"],"tgt_surp"),
    ("tgt","log_TrtT","segment",["tgt_seg_len","tgt_freq","tgt_posq"],"tgt_surp"),
    ("tgt","log_Dur","token",["tgt_seg_len","tgt_freq","tgt_posq"],"tgt_surp"),
    ("tgt","log_Dur","segment",["tgt_seg_len","tgt_freq","tgt_posq"],"tgt_surp"),
]

def _try(model, maxiter):
    best = None
    for m in ("lbfgs","bfgs","cg"):
        r = model.fit(reml=False, method=m, maxiter=maxiter, disp=False)
        if best is None or float(r.llf) > float(best.llf): best = r
        if bool(getattr(r,"converged",False)): return float(r.llf), int(r.nobs), True
    return float(best.llf), int(best.nobs), False

def _fit(w, formula, vc, maxiter):
    m = smf.mixedlm(formula, data=w, groups=w["_g"], re_formula="0", vc_formula=vc)
    return _try(m, maxiter)

def main():
    from run_mixed_effects_yiyu_analysis import load_data, filter_group
    data = load_data(Path(__file__).resolve().parent/"attention-norm-yiyu-positive-dur")
    rows = []
    for side, label, level, ctrls, pred in MODELS:
        w = filter_group(data[side][level], "all").dropna(subset=ctrls+[pred,label,"lang_pair","translator_id","sentence_item_id"]).copy()
        w["_g"] = "all"
        # Bypass pandas StringDtype: convert string cols to object
        for c in ["translator_id","sentence_item_id","lang_pair","_g"]:
            w[c] = w[c].astype("object")
        if len(w) < 40 or w["translator_id"].nunique() < 2: continue

        t0 = time.time()
        fx = " + ".join(ctrls + ["C(lang_pair)"])
        fx_p = " + ".join(ctrls + [pred, "C(lang_pair)"])

        # base (controls only, RI)
        bl, bn, _ = _fit(w, f"{label} ~ {fx}", {"p":"0+C(translator_id)","i":"0+C(sentence_item_id)"}, 300)

        # RI-only augmented
        rl, rn, _ = _fit(w, f"{label} ~ {fx_p}", {"p":"0+C(translator_id)","i":"0+C(sentence_item_id)"}, 300)

        # Tier 1: maximal
        ml, mn, mc = _fit(w, f"{label} ~ {fx_p}", {
            "p":"1+C(translator_id)", f"p_{pred}":f"0+C(translator_id):{pred}",
            "i":"0+C(sentence_item_id)",
        }, 300)

        tier = "maximal" if mc else "uncorrelated"  # vc terms are already independent
        best_l, best_n = (ml, mn) if mc else (rl, rn)
        if not mc: tier = "RI-only"; best_l, best_n = rl, rn

        d = (best_l - bl)/max(best_n,1)
        rd = (rl - bl)/max(rn,1)
        print(f"  {label:10s} {level:8s} {pred:15s} tier={tier:13s} ΔLL/n={d:+.5f} RI={rd:+.5f} diff={d-rd:+.5f} conv={mc} {time.time()-t0:.1f}s")
        rows.append({"side":side,"label":label,"level":level,"predictor":pred,"tier":tier,"converged":mc,"n_obs":best_n,"n_subj":w["translator_id"].nunique(),"delta_ll_per_obs":d,"delta_ll_per_obs_ri":rd,"delta_diff":d-rd})

    out = Path(__file__).resolve().parent/"figures_yiyu_positive_dur_mixed"/"sensitivity_random_slope.tsv"
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    print(f"\nWrote {out}")
    print(pd.DataFrame(rows)[["label","level","predictor","tier","delta_ll_per_obs","delta_ll_per_obs_ri"]].to_string(index=False))

if __name__ == "__main__": main()
