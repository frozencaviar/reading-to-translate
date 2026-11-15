#!/usr/bin/env Rscript
# lme4 interaction models: predictor × task_type and predictor × lang_pair
# Output: table_mixed_interaction_all_tests.tsv, figure_interaction_mixed.tsv (same schema as Python)
library(lme4)
suppressMessages(library(dplyr))

args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) > 0) args[1] else "figures_yiyu_positive_dur_mixed_lme4"
outdir <- file.path(data_dir)
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

cat("Loading data...\n")
d <- read.csv("model-input/all_model_data.csv", check.names = FALSE)
for (v in c("lang_pair","translator_id","sentence_item_id")) {
  d[[v]] <- factor(d[[v]])
}

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

bh_adjust <- function(pvals) {
  n <- length(pvals); o <- order(pvals); q <- rep(NA, n); running <- 1
  for (i in n:1) { running <- min(running, pvals[o[i]] * n / i); q[o[i]] <- min(running, 1) }; q
}

interactions <- list(
  task = list(var = "task_type", has_cross = TRUE),
  direction = list(var = "lang_pair", has_cross = TRUE)
)

records <- list()
status_rows <- list()

for (int_name in names(interactions)) {
  int_var <- interactions[[int_name]]$var
  cat(sprintf("[lme4-interaction] %s\n", int_name))

  for (side in names(SIDES)) {
    for (level in names(LEVELS)) {
      for (label in LABELS[[side]]) {
        ctrls <- CONTROLS[[side]][[level]]
        preds <- FEATURES[[side]]

        keep <- d[["_side"]] == side & d[["_level"]] == level
        if (int_name == "direction") {
          keep <- keep & !is.na(d[["lang_pair"]]) & d[["task_type"]] != ""
        } else {
          keep <- keep & !is.na(d[["task_type"]]) & d[["task_type"]] != ""
        }

        for (pred in preds) {
          cols <- c(label, ctrls, pred, "lang_pair", "task_type", "translator_id", "sentence_item_id")
          sub <- d[keep, cols]
          sub <- na.omit(sub)
          sub$lang_pair <- droplevels(sub$lang_pair)
          sub$task_type <- factor(sub$task_type)

          n_lp <- length(levels(sub$lang_pair))
          n_tt <- length(levels(sub$task_type))
          nsubj <- length(unique(sub$translator_id))
          nitem <- length(unique(sub$sentence_item_id))

          if (nrow(sub) < 40 || nsubj < 2 || nitem < 2) {
            status_rows[[length(status_rows) + 1]] <- list(
              interaction = int_name, side = side, label = label, level = level,
              predictor = pred, status = "insufficient",
              n_rows = nrow(sub), n_participants = nsubj, n_items = nitem
            )
            next
          }

          # Build fixed effects
          fx_ctrl <- paste(ctrls, collapse = " + ")
          re_rs <- sprintf("(1 + %s || translator_id) + (1 | sentence_item_id)", pred)
          re_ri <- "(1 | translator_id) + (1 | sentence_item_id)"

          # Only include factors that vary
          group_terms <- c()
          if (n_lp > 1) group_terms <- c(group_terms, "lang_pair")
          if (n_tt > 1) group_terms <- c(group_terms, "task_type")

          # Additive: controls + predictor + group terms
          fx_add <- paste(c(ctrls, pred, group_terms), collapse = " + ")
          # Interaction: controls + predictor * int_var + other group term
          if (int_name == "task" && n_tt > 1) {
            fx_int <- paste(c(ctrls, sprintf("%s * task_type", pred),
                              if (n_lp > 1) "lang_pair" else c()), collapse = " + ")
          } else if (int_name == "direction" && n_lp > 1) {
            fx_int <- paste(c(ctrls, sprintf("%s * lang_pair", pred),
                              if (n_tt > 1) "task_type" else c()), collapse = " + ")
          } else {
            status_rows[[length(status_rows) + 1]] <- list(
              interaction = int_name, side = side, label = label, level = level,
              predictor = pred, status = "insufficient_factor_levels",
              n_rows = nrow(sub), n_participants = nsubj, n_items = nitem
            )
            next
          }

          # Recovery ladder for the interaction models, parallel to the main
          # analysis but with by-participant slopes only: by-item random slopes for
          # the focal predictor are not added because sentence items are nested
          # within task and direction, so by-item task/direction slopes are not
          # estimable. We start from a correlated by-participant slope, remove the
          # correlation (||) on failure, then fall back to random intercepts only
          # (accepting a singular boundary fit). Additive and interaction models
          # share the random structure so the LRT isolates the interaction term.
          re_steps <- list(
            list(kind = "lme4_corr_slope", re = sprintf("(1 + %s | translator_id) + (1 | sentence_item_id)", pred)),
            list(kind = "lme4_shared_slope", re = re_rs),
            list(kind = "lme4_RI_only",      re = re_ri)
          )
          m_add <- NULL; m_int <- NULL; model_kind <- NA_character_
          for (si in seq_along(re_steps)) {
            re_tier <- re_steps[[si]]$re
            model_kind <- re_steps[[si]]$kind
            f_add <- as.formula(sprintf("%s ~ %s + %s", label, fx_add, re_tier))
            f_int <- as.formula(sprintf("%s ~ %s + %s", label, fx_int, re_tier))
            m_add <- try(lmer(f_add, data = sub, REML = FALSE,
                              control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))), silent = TRUE)
            m_int <- try(lmer(f_int, data = sub, REML = FALSE,
                              control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))), silent = TRUE)
            is_floor <- (si == length(re_steps))
            if (is_floor) {
              if (!conv_failed(m_add) && !conv_failed(m_int)) break
            } else if (fit_ok(m_add) && fit_ok(m_int)) {
              break
            }
          }

          if (conv_failed(m_add) || conv_failed(m_int)) {
            status_rows[[length(status_rows) + 1]] <- list(
              interaction = int_name, side = side, label = label, level = level,
              predictor = pred, status = "nonconverged",
              n_rows = nrow(sub), n_participants = nsubj, n_items = nitem
            )
            next
          }

          delta <- as.numeric(logLik(m_int) - logLik(m_add))
          n <- nobs(m_int)
          lr_add_df <- length(fixef(m_add))
          lr_int_df <- length(fixef(m_int))
          df_lrt <- lr_int_df - lr_add_df
          lr_stat <- max(0, 2 * delta)
          p_raw <- pchisq(lr_stat, df = df_lrt, lower.tail = FALSE)

          records[[length(records) + 1]] <- list(
            status = "ok",
            interaction = int_name,
            predictor = pred,
            feature = pred,
            metric = delta / n,
            p_raw = p_raw,
            n_participants = nsubj, n_items = nitem, n_rows = n,
            llf_additive = as.numeric(logLik(m_add)),
            llf_interaction = as.numeric(logLik(m_int)),
            lr_stat = lr_stat, df_lrt = df_lrt,
            additive_model_kind = model_kind,
            interaction_model_kind = model_kind,
            additive_converged = TRUE, interaction_converged = TRUE,
            additive_singular = isSingular(m_add),
            interaction_singular = isSingular(m_int),
            side = side, label = label, level = level
          )
        }
      }
    }
  }
}

tbl <- do.call(rbind, lapply(records, as.data.frame, stringsAsFactors = FALSE))
if (nrow(tbl) > 0) {
  tbl$q_bh <- bh_adjust(tbl$p_raw)
  tbl$sig_bh <- tbl$q_bh < 0.05
}

write.table(tbl, file.path(outdir, "table_mixed_interaction_all_tests.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)

# Also write the simplified version for the plot script
fig_cols <- c("interaction","side","label","level","predictor","feature","metric","p_raw","n_participants","n_items","n_rows","q_bh","sig_bh")
if (nrow(tbl) > 0) {
  fig <- tbl[, intersect(fig_cols, names(tbl))]
  write.table(fig, file.path(outdir, "figure_interaction_mixed.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
}

st <- do.call(rbind, lapply(status_rows, as.data.frame, stringsAsFactors = FALSE))
write.table(st, file.path(outdir, "table_mixed_interaction_fit_status.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)

cat(sprintf("[lme4-interaction] wrote %d records, %d status rows\n", nrow(tbl), nrow(st)))
