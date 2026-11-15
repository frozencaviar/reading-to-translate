#!/usr/bin/env Rscript
# Attention shrinkage under matched observations and matched random-effects tier.
# Output:
# - att-shrinkage.tsv: maximal-only aggregate for the main figure
# - table_att_shrinkage_paired.tsv: full per-feature detail, including fallbacks
# - table_att_shrinkage_fit_status.tsv: failed/fallback tier audit
library(lme4)

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) > 0) args[1] else "figures_yiyu_positive_dur_mixed_lme4"
dir.create(file.path(outdir), showWarnings = FALSE, recursive = TRUE)

cat("Loading...\n")
d <- read.csv("model-input/all_model_data.csv", check.names = FALSE)
for (v in c("lang_pair","translator_id","sentence_item_id")) d[[v]] <- factor(d[[v]])

SIDES  <- list(src = "src", tgt = "tgt")
LEVELS <- list(token = "token", segment = "segment")
LABELS <- list(src = c("log_TrtS"), tgt = c("log_TrtT", "log_Dur"))
CONTROLS <- list(
  src = list(token = c("src_seg_len","src_freq","src_posq"),
             segment = c("src_seg_len","src_freq","src_posq")),
  tgt = list(token = c("tgt_seg_len","tgt_freq","tgt_posq"),
             segment = c("tgt_seg_len","tgt_freq","tgt_posq"))
)

# Attention features per side (excluding the surprisal features themselves)
ATTN_SRC <- c("x2x","x2cont","x2eos","x_ent","cont2x","tgtseq2x")
ATTN_TGT <- c("eos_xatt","y2srcseq_ent","y2y","y2cont","y_ent")
ATTN <- list(src = ATTN_SRC, tgt = ATTN_TGT)

# Surprisal baselines
SURP <- list(LM = list(src = "src_surp", tgt = "tgt_surp"),
             NMT = list(tgt = "xent"))

CELL_NAMES <- list(
  `src_log_TrtS_token`    = "Source reading (token)",
  `src_log_TrtS_segment`  = "Source reading (segment)",
  `tgt_log_TrtT_token`    = "Target reading (token)",
  `tgt_log_TrtT_segment`  = "Target reading (segment)",
  `tgt_log_Dur_token`     = "Target duration (token)",
  `tgt_log_Dur_segment`   = "Target duration (segment)"
)

conv_failed <- function(m) {
  if (inherits(m, "try-error")) return(TRUE)
  if (!isTRUE(m@optinfo$conv$opt == 0)) return(TRUE)
  msgs <- m@optinfo$conv$lme4$messages
  if (is.null(msgs)) return(FALSE)
  non_singular <- msgs[!grepl("singular|boundary", msgs, ignore.case = TRUE)]
  length(non_singular) > 0
}
fit_ok <- function(m) {
  !conv_failed(m) && !(!inherits(m, "try-error") && isSingular(m))
}
# At the RI-only floor a singular boundary fit is accepted (no simpler structure).
fit_usable <- function(m, tier_str) {
  if (tier_str == "RI-only") !conv_failed(m) else fit_ok(m)
}

fit_model <- function(sub, label, fixed_terms, pred, tier_str) {
  lvl <- length(unique(droplevels(sub$lang_pair)))
  fx_ctrl <- unique(fixed_terms)
  if (lvl > 1 && !"lang_pair" %in% fx_ctrl) fx_ctrl <- c(fx_ctrl, "lang_pair")
  fx <- paste(fx_ctrl, collapse = " + ")
  if (tier_str == "correlated_maximal") {
    re <- sprintf("(1 + %s | translator_id) + (1 + %s | sentence_item_id)", pred, pred)
  } else if (tier_str == "maximal") {
    re <- sprintf("(1 + %s || translator_id) + (1 + %s || sentence_item_id)", pred, pred)
  } else if (tier_str == "by_participant") {
    re <- sprintf("(1 + %s || translator_id) + (1 | sentence_item_id)", pred)
  } else {
    re <- "(1 | translator_id) + (1 | sentence_item_id)"
  }
  f <- as.formula(sprintf("%s ~ %s + %s", label, fx, re))
	  try(lmer(f, data = sub, REML = FALSE,
	           control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))), silent = TRUE)
}

fit_status <- function(m) {
  if (inherits(m, "try-error")) return("failed")
  if (!is.null(m@optinfo$conv$lme4$messages)) return("nonconverged")
  if (!isTRUE(m@optinfo$conv$opt == 0)) return("nonzero_optimizer_code")
  "ok"
}

singular_flag <- function(m) {
  if (inherits(m, "try-error")) return(NA)
  isSingular(m)
}

lrt_p <- function(delta) {
  pchisq(max(0, 2 * delta), df = 1, lower.tail = FALSE)
}

fit_comparison <- function(sub, label, base_terms, aug_terms, attn, tier) {
  m_base <- fit_model(sub, label, base_terms, attn, tier)
  m_aug <- fit_model(sub, label, aug_terms, attn, tier)
  ok <- fit_usable(m_base, tier) && fit_usable(m_aug, tier)
  delta <- NA_real_
  metric <- NA_real_
  p_raw <- NA_real_
  llf_base <- NA_real_
  llf_aug <- NA_real_
  n <- NA_integer_
  if (ok) {
    delta <- as.numeric(logLik(m_aug) - logLik(m_base))
    n <- nobs(m_aug)
    metric <- delta / n
    p_raw <- lrt_p(delta)
    llf_base <- as.numeric(logLik(m_base))
    llf_aug <- as.numeric(logLik(m_aug))
  }
  list(
    ok = ok,
    metric = metric,
    p_raw = p_raw,
    llf_base = llf_base,
    llf_aug = llf_aug,
    lr_stat = if (is.na(delta)) NA_real_ else max(0, 2 * delta),
    n = n,
    base_status = fit_status(m_base),
    aug_status = fit_status(m_aug),
    base_singular = singular_flag(m_base),
    aug_singular = singular_flag(m_aug)
  )
}

detail_rows <- list()
status_rows <- list()

for (side in names(SIDES)) {
  for (level in names(LEVELS)) {
    for (label in LABELS[[side]]) {
      ctrls <- CONTROLS[[side]][[level]]
      cell_key <- sprintf("%s_%s_%s", side, label, level)
      cell_name <- CELL_NAMES[[cell_key]]
      if (is.null(cell_name)) next

      attn_feats <- ATTN[[side]]

      for (attn in attn_feats) {
        # LM panel: src_surp for source, tgt_surp for target
        surp_lm <- SURP$LM[[side]]
        # NMT panel: xent for target only
        surp_nmt <- SURP$NMT[[side]]

        keep <- d[["_side"]] == side & d[["_level"]] == level
        cols <- c(label, ctrls, attn, "lang_pair", "translator_id", "sentence_item_id")
        if (!is.null(surp_lm)) cols <- c(cols, surp_lm)
        if (!is.null(surp_nmt)) cols <- c(cols, surp_nmt)
        sub <- d[keep, unique(cols)]; sub <- na.omit(sub)
        nsubj <- length(unique(sub$translator_id))
        nitem <- length(unique(sub$sentence_item_id))
        if (nrow(sub) < 40 || nsubj < 2 || nitem < 2) {
          for (baseline_model in c("LM", if (!is.null(surp_nmt)) "NMT" else NULL)) {
            status_rows[[length(status_rows) + 1]] <- list(
              baseline_model = baseline_model, side = side, label = label, level = level,
              cell = cell_name, feature = attn, tier = NA, status = "insufficient",
              n_rows = nrow(sub), n_participants = nsubj, n_items = nitem,
              before_base_status = NA, before_aug_status = NA,
              after_base_status = NA, after_aug_status = NA,
              selected = FALSE
            )
          }
          next
        }

        baseline_specs <- list()
        if (!is.null(surp_lm)) baseline_specs[["LM"]] <- surp_lm
        if (!is.null(surp_nmt)) baseline_specs[["NMT"]] <- surp_nmt

        for (baseline_model in names(baseline_specs)) {
          surp <- baseline_specs[[baseline_model]]
          selected <- FALSE

          for (tier in c("correlated_maximal", "maximal", "by_participant", "RI-only")) {
            before <- fit_comparison(sub, label, ctrls, c(ctrls, attn), attn, tier)
            after <- fit_comparison(sub, label, c(ctrls, surp), c(ctrls, surp, attn), attn, tier)
            ok <- before$ok && after$ok
            status <- if (ok) "ok" else "nonconverged"

            status_rows[[length(status_rows) + 1]] <- list(
              baseline_model = baseline_model, side = side, label = label, level = level,
              cell = cell_name, feature = attn, tier = tier, status = status,
              n_rows = nrow(sub), n_participants = nsubj, n_items = nitem,
              before_base_status = before$base_status,
              before_aug_status = before$aug_status,
              after_base_status = after$base_status,
              after_aug_status = after$aug_status,
              selected = ok
            )

            if (ok) {
              detail_rows[[length(detail_rows) + 1]] <- list(
                baseline_model = baseline_model, side = side, label = label, level = level,
                cell = cell_name, feature = attn, tier = tier,
                n_rows = before$n, n_participants = nsubj, n_items = nitem,
                before_metric = before$metric,
                after_metric = after$metric,
                shrinkage_metric = before$metric - after$metric,
                retention_ratio = if (is.na(before$metric) || abs(before$metric) < 1e-12) NA_real_ else after$metric / before$metric,
                before_p_raw = before$p_raw,
                after_p_raw = after$p_raw,
                llf_before_base = before$llf_base,
                llf_before_aug = before$llf_aug,
                llf_after_base = after$llf_base,
                llf_after_aug = after$llf_aug,
                lr_before = before$lr_stat,
                lr_after = after$lr_stat,
                before_base_singular = before$base_singular,
                before_aug_singular = before$aug_singular,
                after_base_singular = after$base_singular,
                after_aug_singular = after$aug_singular
              )
              selected <- TRUE
              break
            }
          }

          if (!selected) {
            status_rows[[length(status_rows) + 1]] <- list(
              baseline_model = baseline_model, side = side, label = label, level = level,
              cell = cell_name, feature = attn, tier = NA, status = "failed_all_tiers",
              n_rows = nrow(sub), n_participants = nsubj, n_items = nitem,
              before_base_status = NA, before_aug_status = NA,
              after_base_status = NA, after_aug_status = NA,
              selected = FALSE
            )
          }
        }
      }
    }
  }
}

detail_tbl <- do.call(rbind, lapply(detail_rows, as.data.frame, stringsAsFactors = FALSE))
write.table(detail_tbl, file.path(outdir, "table_att_shrinkage_paired.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)

status_tbl <- do.call(rbind, lapply(status_rows, as.data.frame, stringsAsFactors = FALSE))
write.table(status_tbl, file.path(outdir, "table_att_shrinkage_fit_status.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)

# ---- Aggregate to shrinkage format ----
# Main figure aggregates the features whose before/after pair both reached a
# maximal (by-participant + by-item slope) structure, correlated or uncorrelated,
# to avoid mixing shrinkage values across reduced fallback tiers.
agg <- detail_tbl[detail_tbl$tier %in% c("correlated_maximal", "maximal"), ]
shrink_tbl <- aggregate(cbind(before_metric, after_metric) ~ baseline_model + side + label + level + cell,
                        data = agg, FUN = mean, na.rm = TRUE)
names(shrink_tbl)[names(shrink_tbl) == "before_metric"] <- "control_only"
names(shrink_tbl)[names(shrink_tbl) == "after_metric"] <- "control_plus_surprisal"
feature_counts <- aggregate(feature ~ baseline_model + side + label + level + cell,
                            data = agg, FUN = length)
names(feature_counts)[names(feature_counts) == "feature"] <- "n_features"
shrink_tbl <- merge(shrink_tbl, feature_counts,
                    by = c("baseline_model","side","label","level","cell"), all.x = TRUE)
shrink_tbl$tier_scope <- "maximal_only"
shrink_tbl$difference <- shrink_tbl$control_only - shrink_tbl$control_plus_surprisal
shrink_tbl <- shrink_tbl[order(shrink_tbl$baseline_model, shrink_tbl$side, shrink_tbl$label, shrink_tbl$level), ]
shrink_tbl <- shrink_tbl[, c("baseline_model","side","label","level","cell","tier_scope","n_features","control_only","control_plus_surprisal","difference")]

write.table(shrink_tbl, file.path(outdir, "att-shrinkage.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
cat(sprintf("Wrote %d detail rows, %d status rows, %d maximal-only aggregate rows\n",
            nrow(detail_tbl), nrow(status_tbl), nrow(shrink_tbl)))
