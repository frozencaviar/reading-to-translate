# Reading to Translate or Translating to Read?

## Cite this study

Zhang, Y., Yao, X., & Li, D. (2026). Reading to Translate or Translating to Read? Modeling Translators’ Eye Movements with Multilingual Pre-Trained Models. *Journal of Eye Movement Research, 19*(3), Article 66. [https://doi.org/10.3390/jemr19030066](https://doi.org/10.3390/jemr19030066)

## Quickstart

After installing the dependencies below, supply the directory containing your session tables and run the cross-validated analysis:

```bash
bash extract.sh "/path/to/Tables"
python reproduce_yiyu_analysis.py --analysis-only
```

This code reproduces the feature-extraction and statistical-modeling workflow for the English–Chinese translation and post-editing study. Supply your own authorized CRITT-format tables. The archive contains code only. Run the commands below from the extracted package directory.

## Installation

Use Python 3.10 or 3.11 and Bash. The dependency ranges provide a starting environment. The original run did not supply a version lock. For the R analyses, install R with `Rscript` on your PATH.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
Rscript -e 'install.packages(c("lme4","dplyr"), repos="https://cloud.r-project.org")'
```

## 1. Extract features from the session tables

Point the command at the directory containing matching UTF-8, tab-delimited `.st` (source token), `.tt` (target token) and `.sg` (sentence) tables. Each session needs the same basename across all three files. Preserve the study and session metadata, language codes, segment IDs, token alignments and timing columns. The modeling scripts infer task from the second underscore-separated field in the session ID (T for translation, P for post-editing).

```bash
bash extract.sh "/path/to/Tables"
```

The wrapper builds a local tokenized session manifest, then computes all sessions from scratch. The historical workflow reused previously extracted sessions. Both use normalized attention, maximum over heads and batch size 8. The first run downloads the Hugging Face checkpoints `facebook/nllb-200-distilled-600M` and `sberbank-ai/mGPT`, unless cached. The extractor supports CUDA, Apple MPS and CPU.

Check the generated `attention-norm-yiyu-positive-dur/yiyu-features-*.tsv` files for session coverage before modeling. The two terms in each filename give the source and target units, respectively. Token–segment uses the source token table, segment–token the target token table, and sentence–sentence the sentence table. The TSVs include lexical items and session identifiers. Keep these and the generated `data/sessions.json` with the private inputs.

## 2. Fit the statistical models

If you already have all three compatible feature TSVs, place them in the directory above and start here.

| Analysis | Command | Output directory |
| --- | --- | --- |
| Cross-validated OLS, changes in log likelihood and permutation inference | `python reproduce_yiyu_analysis.py --analysis-only` | `figures_yiyu_positive_dur/` |
| Mixed-effects model comparisons | `python run_mixed_effects_yiyu_analysis.py` | See the script's `--outdir` option |
| Mixed-effects interactions | `python run_mixed_interaction_yiyu_analysis.py` | See `--help` for output options |
| Two-part duration model (Python preparation and R fitting) | `python run_two_part_glmm_dur_analysis.py` | `figures_yiyu_positive_dur_twopart_glmm/` |

The reading outcomes are log source and target reading times (`log_TrtS`, `log_TrtT`). The positive-duration analysis uses target production durations of at least 20 ms. The two-part model separately estimates the probability of editing and log duration conditional on meeting the positive-duration threshold. Install `wordfreq` for the lexical-frequency control. Some mixed-model scripts fall back to corpus frequencies if that package is unavailable, which can change the estimates.

For the R mixed-effects analyses, prepare the shared model input once, then run the desired fits:

```bash
python prepare_lme4_input.py
Rscript run_lme4_analysis.R
Rscript run_lme4_interaction.R
Rscript run_lme4_coefficients.R
Rscript run_lme4_att_shrinkage.R
```

The preparation script uses the Python mixed-model preprocessing and writes `model-input/all_model_data.csv` with explicit side and level columns. Review fitting status and convergence alongside model comparisons.

## Optional post-editing draft analysis

Supply a private UTF-8 JSON object keyed by condition (`P1` and `P2`). Each condition contains `src_lang` and `tgt_lang` (en or zh), plus `source_segments` and `target_segments`. Each segment is an array of token strings. Pair segments in their original order so the one-based segment IDs match the session tables. Use the initial MT drafts for this analysis.

```bash
python compute_mt_draft_features.py "/path/to/private-drafts.json"
python merge_mt_draft_to_tsv.py
python analyze_mt_draft.py
```

The merge writes feature files with the suffix `_with_mt_draft`. The final script compares draft features for edited and unedited segments using a permutation test.

## Additional analyses in the study project

The project includes the following supplementary and exploratory analyses. These entries describe the project scripts and do not imply that every analysis was reported in the paper.

| Question | Entry point and inputs |
| --- | --- |
| Do NMT features predict segment-level keystroke measures? | After extraction, run `python keylogging_segment_analysis.py --table-dir "/path/to/Tables"`. It joins sentence features to the private .sg tables and models PreGap, TD1000, TB1000 and Del, with Nedit as a robustness check. |
| Which predictors are associated with reaching 20 ms of production time? | `python stage1_dur_positive_analysis.py` fits exploratory pooled logistic models and tests predictor-by-task and predictor-by-direction interactions. |
| How sensitive are six core fits to participant random slopes? | `python run_random_slope_sensitivity.py` uses the feature TSVs. |
| How do model comparisons vary across participants? | After the two-part fit, use the commands below for participant-level deltas and leave-one-participant-out sign-flip inference. |
| How much does attention add after controlling for surprisal? | After the Python mixed-model fit, run `python export_mixed_att_shrinkage.py`. Companion scripts `export_mixed_aic.py`, `export_mixed_vif.py` and `export_mixed_effects_coefficients.py` report AIC gains, collinearity diagnostics and coefficients. |
| How stable are the main estimates, and how do they vary by subgroup? | The main OLS script writes effect-size, repeated-split stability, multiple-testing, task/direction and part-of-speech summaries. |

```bash
out=figures_yiyu_positive_dur_twopart_glmm
Rscript run_two_part_participant_delta.R "$out/table_twopart_prepared_model_data.tsv" "$out/table_twopart_participant_delta.tsv"
python run_two_part_permutation.py --all-tests "$out/table_twopart_glmm_all_tests.tsv" --data-dir attention-norm-yiyu-positive-dur --outdir "$out"
```

## Verification

The release was checked for archive integrity and Python and shell syntax. Numerical reproduction requires the study inputs, model checkpoints and matching software versions.

## Acknowledgments

This package adapts Zheng Wei Lim's implementation for the present English–Chinese translation and post-editing study. The NMT feature-extraction code in `src/` derives from Lim's repository, including the routines for reading CRITT tables and computing surprisal and attention features. The cross-validated regression workflow in `reproduce_yiyu_analysis.py` follows the original project's analysis notebook. [Original code repository](https://github.com/ZhengWeiLim/pred-trans-difficulty-NMT)

The present study adapts session preparation and analysis to English–Chinese translation and post-editing. Its scripts apply a 20 ms minimum to positive target production durations, compare conditions with mixed-effects models, and fit a two-part model for editing probability and duration. For this distribution, the extraction wrapper rebuilds features from user-supplied tables, and the optional MT-draft analysis reads the experimental text from a private JSON file.

When using the feature extractor or its analysis approach, please cite the original paper and acknowledge the source repository. The upstream MIT license and its copyright notice, “Copyright (c) 2023 Zheng Wei Lim,” are preserved in [LICENSE](LICENSE).

Lim, Z. W., Vylomova, E., Kemp, C., & Cohn, T. (2024). Predicting human translation difficulty with neural machine translation. *Transactions of the Association for Computational Linguistics, 12*, 1479–1496. [DOI](https://doi.org/10.1162/tacl_a_00714) · [ACL Anthology](https://aclanthology.org/2024.tacl-1.81/)
