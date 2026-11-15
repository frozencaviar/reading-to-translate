#!/usr/bin/env Rscript
# Export coefficient estimates with 95% CI from lme4 models
# Output schema: side, label, level, focus_feature, predictor, mean_coef, ci95_halfwidth
library(lme4)

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) > 0) args[1] else "figures_yiyu_positive_dur_mixed_lme4"
dir.create(file.path(outdir), showWarnings = FALSE, recursive = TRUE)

cat("Loading data...\n")
d <- read.csv("model-input/all_model_data.csv", check.names = FALSE)
for (v in c("lang_pair","translator_id","sentence_item_id")) d[[v]] <- factor(d[[v]])

SIDES <- list(src = "src", tgt = "tgt")
LEVELS <- list(token = "token", segment = "segment")
LABELS <- list(src = c("log_TrtS"), tgt = c("log_TrtT", "log_Dur"))
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
fit_ok <- function(m) {
  !conv_failed(m) && !is_singular(m)
}

coef_rows <- list()

for (side in names(SIDES)) {
  for (level in names(LEVELS)) {
    for (label in LABELS[[side]]) {
      ctrls <- CONTROLS[[side]][[level]]
      preds <- FEATURES[[side]]
      keep <- d[["_side"]] == side & d[["_level"]] == level
      sub_base <- d[keep, c(label, ctrls, "lang_pair", "translator_id", "sentence_item_id")]
      sub_base <- na.omit(sub_base)

      for (pred in preds) {
        cols <- c(label, ctrls, pred, "lang_pair", "translator_id", "sentence_item_id")
        sub <- d[keep, cols]; sub <- na.omit(sub)
        if (nrow(sub) < 40) next

        lvl <- length(unique(droplevels(sub$lang_pair)))
        fx_ctrl <- c(ctrls, if (lvl > 1) "lang_pair" else c(), pred)
        fx <- paste(fx_ctrl, collapse = " + ")
        # Same recovery ladder as the main analysis: correlated maximal -> ||
        # -> by-participant slope -> RI-only floor (accepting a singular floor fit).
        re_ladder <- c(
          sprintf("(1 + %s | translator_id) + (1 + %s | sentence_item_id)", pred, pred),
          sprintf("(1 + %s || translator_id) + (1 + %s || sentence_item_id)", pred, pred),
          sprintf("(1 + %s || translator_id) + (1 | sentence_item_id)", pred),
          "(1 | translator_id) + (1 | sentence_item_id)"
        )
        m <- NULL
        for (ri in seq_along(re_ladder)) {
          f <- as.formula(sprintf("%s ~ %s + %s", label, fx, re_ladder[[ri]]))
          m <- try(lmer(f, data = sub, REML = FALSE,
                        control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))), silent = TRUE)
          is_floor <- (ri == length(re_ladder))
          if (is_floor) {
            if (!conv_failed(m)) break       # floor: accept singular boundary fit
          } else if (fit_ok(m)) {
            break
          }
        }
        # Accept a converged RI-only floor even if singular (valid for fixed-effect CIs).
        if (conv_failed(m)) next

        fe <- summary(m)$coefficients
        ci <- try(confint(m, method = "Wald"), silent = TRUE)
        if (inherits(ci, "try-error")) next

        for (term in rownames(fe)) {
          if (!term %in% rownames(ci)) next
          coef_rows[[length(coef_rows) + 1]] <- list(
            side = side, label = label, level = level,
            focus_feature = pred, predictor = term,
            mean_coef = fe[term, "Estimate"],
            ci95_halfwidth = (ci[term, 2] - ci[term, 1]) / 2
          )
        }
      }
    }
  }
}

tbl <- do.call(rbind, lapply(coef_rows, as.data.frame, stringsAsFactors = FALSE))
# Add plot significance: CI not crossing zero
tbl$sig_ci <- abs(tbl$mean_coef) > tbl$ci95_halfwidth
# For compatibility with plot_coef_heatmap_topjournal.py (needs sig_bh_plot)
tbl$sig_bh_plot <- tbl$sig_ci
write.table(tbl, file.path(outdir, "figure2_coefficients.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
write.table(tbl, file.path(outdir, "figure2_coefficients_with_fdr.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
cat(sprintf("Wrote %d coefficient rows\n", nrow(tbl)))
