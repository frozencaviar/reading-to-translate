#!/usr/bin/env Rscript
# Full lme4 pipeline: replicates run_mixed_effects_yiyu_analysis.py output schema
# Output: identical TSV files for downstream plot scripts
library(lme4)
suppressMessages(library(dplyr))

args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) > 0) args[1] else "figures_yiyu_positive_dur_mixed_lme4"
outdir <- file.path(data_dir)
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# ---- Load data ----
cat("Loading data...\n")
d <- read.csv("model-input/all_model_data.csv", check.names = FALSE)
for (v in c("lang_pair","translator_id","sentence_item_id")) {
  d[[v]] <- factor(d[[v]])
}

# ---- Configuration (mirrors Python) ----
SIDES <- list(src = "src", tgt = "tgt")
LEVELS <- list(token = "token", segment = "segment")
LABELS <- list(src = c("log_TrtS"), tgt = c("log_TrtT", "log_Dur"))
DUR_MIN_MS <- 20

FEATURES <- list(
  src = c("src_surp","x2x","x2cont","x2eos","x_ent","cont2x","tgtseq2x"),
  tgt = c("tgt_surp","xent","eos_xatt","y2srcseq_ent","y2y","y2cont","y_ent")
)
CONTROLS <- list(
  src = list(token = c("src_seg_len","src_freq","src_posq"),
             segment = c("src_seg_len","src_freq","src_posq")),
  tgt = list(token = c("tgt_seg_len","tgt_freq","tgt_posq"),
             segment = c("tgt_seg_len","tgt_freq","tgt_posq"))
)

GROUPS <- list(
  `all`                = expression(TRUE),
  `all_translating`    = expression(task_type == "translating"),
  `all_post-editing`   = expression(task_type == "post-editing"),
  `all_en-zh`          = expression(lang_pair == "en-zh"),
  `all_zh-en`          = expression(lang_pair == "zh-en"),
  `translating_en-zh`  = expression(lang_pair == "en-zh" & task_type == "translating"),
  `translating_zh-en`  = expression(lang_pair == "zh-en" & task_type == "translating"),
  `post-editing_en-zh` = expression(lang_pair == "en-zh" & task_type == "post-editing"),
  `post-editing_zh-en` = expression(lang_pair == "zh-en" & task_type == "post-editing")
)

# ---- Fit pair with shared random structure ----
# A "boundary (singular) fit" is a valid MLE for fixed-effect inference. Following
# Barr et al. (2013), we reduce the random structure when the maximal/slope models
# are singular or fail to converge, but at the RI-only floor we accept a singular
# boundary fit rather than discard the cell, since there is no simpler structure to
# reduce to and the fixed-effect likelihood-ratio test remains valid. Genuine
# convergence failures (optimizer code != 0, non-singular warnings) are never accepted.
conv_failed <- function(m) {
  if (inherits(m, "try-error")) return(TRUE)
  if (!isTRUE(m@optinfo$conv$opt == 0)) return(TRUE)
  msgs <- m@optinfo$conv$lme4$messages
  if (is.null(msgs)) return(FALSE)
  non_singular <- msgs[!grepl("singular|boundary", msgs, ignore.case = TRUE)]
  length(non_singular) > 0
}
is_singular <- function(m) {
  !inherits(m, "try-error") && isSingular(m)
}
# Strict check (converged AND non-singular): drives reduction at maximal/slope tiers.
fit_ok <- function(m) {
  !conv_failed(m) && !is_singular(m)
}

fit_pair <- function(sub, label, controls, predictor, maxiter = 200000) {
  # Build fixed effects: only include lang_pair if it varies
  lvl <- length(unique(droplevels(sub$lang_pair)))
  fx_ctrl <- controls
  if (lvl > 1) fx_ctrl <- c(fx_ctrl, "lang_pair")
  fx_base <- paste(fx_ctrl, collapse = " + ")
  fx_aug  <- paste(c(fx_ctrl, predictor), collapse = " + ")

  # Random-effects recovery ladder (Barr et al., 2013): start from the correlated
  # maximal structure; if it fails to converge or is singular, remove the random
  # correlations (||); then drop the by-item slope; finally fall back to random
  # intercepts only. base and augmented models always share the same structure,
  # so the likelihood-ratio test isolates the focal fixed effect (df = 1).
  re_ladder <- list(
    list(tier = "correlated_maximal",
         re = sprintf("(1 + %s | translator_id) + (1 + %s | sentence_item_id)", predictor, predictor)),
    list(tier = "random_slope_maximal",
         re = sprintf("(1 + %s || translator_id) + (1 + %s || sentence_item_id)", predictor, predictor)),
    list(tier = "random_slope",
         re = sprintf("(1 + %s || translator_id) + (1 | sentence_item_id)", predictor)),
    list(tier = "RI-only",
         re = "(1 | translator_id) + (1 | sentence_item_id)")
  )

  m_base <- NULL; m_aug <- NULL; tier <- NA_character_
  converged <- FALSE; base_singular <- NA; pred_singular <- NA
  for (step in re_ladder) {
    f_base <- as.formula(sprintf("%s ~ %s + %s", label, fx_base, step$re))
    f_aug  <- as.formula(sprintf("%s ~ %s + %s", label, fx_aug, step$re))
    m_base <- try(lmer(f_base, data = sub, REML = FALSE,
                       control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = maxiter))), silent = TRUE)
    m_aug  <- try(lmer(f_aug, data = sub, REML = FALSE,
                       control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = maxiter))), silent = TRUE)
    tier <- step$tier
    if (step$tier == "RI-only") {
      # Floor: accept a singular boundary fit (no simpler structure to reduce to).
      converged <- !conv_failed(m_base) && !conv_failed(m_aug)
    } else {
      converged <- fit_ok(m_base) && fit_ok(m_aug)
    }
    base_singular <- is_singular(m_base)
    pred_singular <- is_singular(m_aug)
    if (converged) break
  }

  if (!converged) return(list(ok = FALSE, tier = tier))

  delta <- as.numeric(logLik(m_aug) - logLik(m_base))
  n <- nobs(m_aug)
  lr_stat <- max(0, 2 * delta)
  p_raw <- pchisq(lr_stat, df = 1, lower.tail = FALSE)

  list(
    ok = TRUE,
    tier = tier,
    metric = delta / n,
    p_raw = p_raw,
    n_participants = length(unique(sub$translator_id)),
    n_items = length(unique(sub$sentence_item_id)),
    n_rows = n,
    llf_control = as.numeric(logLik(m_base)),
    llf_predictor = as.numeric(logLik(m_aug)),
    lr_stat = lr_stat,
    base_model_kind = tier,
    predictor_model_kind = tier,
    base_converged = TRUE,
    predictor_converged = TRUE,
    base_singular = base_singular,
    predictor_singular = pred_singular
  )
}

# ---- BH-FDR adjustment ----
bh_adjust <- function(pvals) {
  n <- length(pvals)
  o <- order(pvals)
  q <- rep(NA, n)
  running <- 1
  for (i in n:1) {
    running <- min(running, pvals[o[i]] * n / i)
    q[o[i]] <- min(running, 1)
  }
  q
}

# ---- Main loop ----
records <- list()
status_rows <- list()
start_time <- Sys.time()

for (grp_name in names(GROUPS)) {
  cat(sprintf("[lme4] group=%s\n", grp_name))
  grp_expr <- GROUPS[[grp_name]]

  for (side in names(SIDES)) {
    for (level in names(LEVELS)) {
      for (label in LABELS[[side]]) {
        ctrls <- CONTROLS[[side]][[level]]
        preds <- FEATURES[[side]]

        # Filter data (only keep rows non-NA for label+controls, not all predictors)
        keep <- d[["_side"]] == side & d[["_level"]] == level & eval(grp_expr, d)
        sub_base <- d[keep, c(label, ctrls, "lang_pair", "translator_id", "sentence_item_id")]
        sub_base <- na.omit(sub_base)

        # Figure 5: each predictor vs controls
        for (pred in preds) {
          cols <- c(label, ctrls, pred, "lang_pair", "translator_id", "sentence_item_id")
          sub <- d[keep, cols]
          sub <- na.omit(sub)
          nsubj <- length(unique(sub$translator_id))
          nitem <- length(unique(sub$sentence_item_id))
          if (nrow(sub) < 40 || nsubj < 2 || nitem < 2) {
            status_rows[[length(status_rows) + 1]] <- list(
              group = grp_name, side = side, level = level, label = label,
              delta = "*ctrl", predictor = pred,
              status = "insufficient", n_rows = nrow(sub),
              n_participants = nsubj, n_items = nitem
            )
            next
          }
          res <- fit_pair(sub, label, ctrls, pred)
          if (res$ok) {
            records[[length(records) + 1]] <- c(list(
              group = grp_name, side = side, label = label, level = level,
              delta = "*ctrl", predictor = paste0("control+", pred), feature = pred
            ), res[setdiff(names(res), "ok")])
          } else {
            status_rows[[length(status_rows) + 1]] <- list(
              group = grp_name, side = side, level = level, label = label,
              delta = "*ctrl", predictor = pred,
              status = res$tier, n_rows = nrow(sub),
              n_participants = nsubj, n_items = nitem
            )
          }
        }

        # Figure 3: attention features after surprisal control
        surp_base <- if (side == "src") "src_surp" else "xent"
        attn_preds <- setdiff(preds, if (side == "src") "src_surp" else c("tgt_surp", "xent"))
        for (pred in attn_preds) {
          cols <- c(label, ctrls, surp_base, pred, "lang_pair", "translator_id", "sentence_item_id")
          sub <- d[keep, cols]
          sub <- na.omit(sub)
          nsubj <- length(unique(sub$translator_id))
          nitem <- length(unique(sub$sentence_item_id))
          if (nrow(sub) < 40 || nsubj < 2 || nitem < 2) {
            status_rows[[length(status_rows) + 1]] <- list(
              group = grp_name, side = side, level = level, label = label,
              delta = "*ctrl+surp", predictor = pred,
              status = "insufficient", n_rows = nrow(sub),
              n_participants = nsubj, n_items = nitem
            )
            next
          }
          res <- fit_pair(sub, label, c(ctrls, surp_base), pred)
          if (res$ok) {
            records[[length(records) + 1]] <- c(list(
              group = grp_name, side = side, label = label, level = level,
              delta = "*ctrl+surp", predictor = paste0("control+", pred), feature = pred
            ), res[setdiff(names(res), "ok")])
          } else {
            status_rows[[length(status_rows) + 1]] <- list(
              group = grp_name, side = side, level = level, label = label,
              delta = "*ctrl+surp", predictor = pred,
              status = res$tier, n_rows = nrow(sub),
              n_participants = nsubj, n_items = nitem
            )
          }
        }
      }
    }
  }
  cat(sprintf("[lme4] finished %s after %.0fs\n", grp_name, difftime(Sys.time(), start_time, units = "secs")))
}

# ---- Assemble tables ----
tbl <- do.call(rbind, lapply(records, as.data.frame, stringsAsFactors = FALSE))
tbl$q_bh <- bh_adjust(tbl$p_raw)
tbl$sig_bh <- tbl$q_bh < 0.05

# ---- Write outputs (same schema as Python) ----
base_cols <- c("group","side","label","level","predictor","feature","metric","p_raw","n_participants","q_bh","sig_bh","tier","base_converged","predictor_converged","base_singular","predictor_singular")
fig3_cols <- c("group","side","label","level","delta","predictor","feature","metric","p_raw","n_participants","q_bh","sig_bh","tier","base_converged","predictor_converged","base_singular","predictor_singular")

MAIN_GROUPS <- c("all","all_translating","all_post-editing","all_en-zh","all_zh-en")
SPLIT_GROUPS <- c("translating_en-zh","translating_zh-en","post-editing_en-zh","post-editing_zh-en")

fig5 <- tbl[tbl$delta == "*ctrl", ]
fig3 <- tbl[tbl$delta %in% c("*ctrl", "*ctrl+surp"), ]

fig1 <- fig5[fig5$group == "all", base_cols[base_cols != "group"]]
fig5_main <- fig5[fig5$group %in% MAIN_GROUPS, base_cols]
fig5_split <- fig5[fig5$group %in% SPLIT_GROUPS, base_cols]
fig3_main <- fig3[fig3$group %in% MAIN_GROUPS, fig3_cols]
fig3_split <- fig3[fig3$group %in% SPLIT_GROUPS, fig3_cols]

write.table(fig1, file.path(outdir, "figure1_delta_llh_main.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
write.table(fig5_main, file.path(outdir, "figure5_delta_llh_grouped.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
write.table(fig5_split, file.path(outdir, "figure5_appendix_delta_llh_grouped_split.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
write.table(fig3_main, file.path(outdir, "figure3_delta_llh_att_grouped.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
write.table(fig3_split, file.path(outdir, "figure3_appendix_delta_llh_att_grouped_split.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
write.table(tbl, file.path(outdir, "table_mixed_effects_all_tests.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)

st <- do.call(rbind, lapply(status_rows, as.data.frame, stringsAsFactors = FALSE))
write.table(st, file.path(outdir, "table_mixed_effects_fit_status.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)

meta <- data.frame(
  key = c("model","engine","item_col","frequency_source","metric","p_raw","groups"),
  value = c(
    "lme4 ML, crossed random effects for participant and sentence item (Barr et al., 2013): correlated maximal by-participant and by-item random slopes for the focal predictor, reduced on nonconvergence/singularity by removing random correlations (||), then dropping the by-item slope, then to random intercepts only (singular boundary fit accepted at the random-intercept floor); base and augmented models share the random structure so the LRT isolates the focal fixed effect",
    "lme4 (bobyqa optimizer)",
    "sentence_item_id",
    "wordfreq (Python preprocessing)",
    "(llf_augmented - llf_base) / nobs",
    "likelihood-ratio chi-square test, df=1",
    paste(names(GROUPS), collapse = ",")
  ),
  stringsAsFactors = FALSE
)
write.table(meta, file.path(outdir, "table_mixed_effects_metadata.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)

cat(sprintf("\n[lme4] wrote %d records, %d status rows → %s\n", nrow(tbl), nrow(st), outdir))
