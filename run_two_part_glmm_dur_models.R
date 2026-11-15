#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(lme4))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: run_two_part_glmm_dur_models.R <prepared_tsv> <output_tsv> [maxfun]", call. = FALSE)
}

input_path <- args[[1]]
output_path <- args[[2]]
maxfun <- if (length(args) >= 3) as.numeric(args[[3]]) else 100000

features <- c("tgt_surp", "xent", "eos_xatt", "y2srcseq_ent", "y2y", "y2cont", "y_ent")
controls <- c("tgt_seg_len", "tgt_freq", "tgt_posq")
levels_to_run <- c("token", "segment")
groups <- c(
  "all",
  "all_translating",
  "all_post-editing",
  "all_en-zh",
  "all_zh-en",
  "translating_en-zh",
  "translating_zh-en",
  "post-editing_en-zh",
  "post-editing_zh-en"
)

filter_group <- function(df, group_name) {
  if (group_name == "all") {
    return(df)
  }
  if (group_name == "all_translating") {
    return(df[df$task_type == "translating", , drop = FALSE])
  }
  if (group_name == "all_post-editing") {
    return(df[df$task_type == "post-editing", , drop = FALSE])
  }
  if (group_name == "all_en-zh") {
    return(df[df$lang_pair == "en-zh", , drop = FALSE])
  }
  if (group_name == "all_zh-en") {
    return(df[df$lang_pair == "zh-en", , drop = FALSE])
  }
  if (group_name == "translating_en-zh") {
    return(df[df$task_type == "translating" & df$lang_pair == "en-zh", , drop = FALSE])
  }
  if (group_name == "translating_zh-en") {
    return(df[df$task_type == "translating" & df$lang_pair == "zh-en", , drop = FALSE])
  }
  if (group_name == "post-editing_en-zh") {
    return(df[df$task_type == "post-editing" & df$lang_pair == "en-zh", , drop = FALSE])
  }
  if (group_name == "post-editing_zh-en") {
    return(df[df$task_type == "post-editing" & df$lang_pair == "zh-en", , drop = FALSE])
  }
  stop(sprintf("Unknown group: %s", group_name), call. = FALSE)
}

nonconstant_fixed_terms <- function(df, terms) {
  keep <- c()
  for (term in terms) {
    vals <- df[[term]]
    vals <- vals[!is.na(vals)]
    if (length(unique(vals)) > 1) {
      keep <- c(keep, term)
    }
  }
  keep
}

make_formula <- function(df, outcome, fixed_terms, predictor = NULL, re_struct = "ri", slope_var = NULL) {
  varying_terms <- nonconstant_fixed_terms(df, c("task_type", "lang_pair"))
  rhs <- c(fixed_terms, varying_terms, predictor)
  has_item <- length(unique(df$sentence_item_id)) > 1
  # Random-effects structure. The focal predictor carries by-participant and
  # by-item random slopes at the maximal tier, reduced on nonconvergence/singularity.
  # The slope is driven by slope_var (the focal predictor), which is held FIXED across
  # the base and augmented models so the two share an identical random structure and
  # the likelihood-ratio test isolates the focal fixed effect (df = 1). If slope_var
  # is not supplied, fall back to the predictor argument for backward compatibility.
  slope <- if (!is.null(slope_var)) slope_var else if (!is.null(predictor)) predictor else NULL
  if (re_struct == "correlated_maximal" && !is.null(slope)) {
    re <- if (has_item) {
      sprintf("(1 + %s | translator_id) + (1 + %s | sentence_item_id)", slope, slope)
    } else {
      sprintf("(1 + %s | translator_id)", slope)
    }
  } else if (re_struct == "maximal" && !is.null(slope)) {
    re <- if (has_item) {
      sprintf("(1 + %s || translator_id) + (1 + %s || sentence_item_id)", slope, slope)
    } else {
      sprintf("(1 + %s || translator_id)", slope)
    }
  } else if (re_struct == "by_participant" && !is.null(slope)) {
    re <- if (has_item) {
      sprintf("(1 + %s || translator_id) + (1 | sentence_item_id)", slope)
    } else {
      sprintf("(1 + %s || translator_id)", slope)
    }
  } else {
    re <- "(1 | translator_id)"
    if (has_item) re <- paste(re, "(1 | sentence_item_id)", sep = " + ")
  }
  as.formula(paste(outcome, "~", paste(c(rhs, re), collapse = " + ")))
}

fit_one_model <- function(formula, df, family_name) {
  if (family_name == "binomial") {
    ctrl <- glmerControl(
      optimizer = "bobyqa",
      optCtrl = list(maxfun = maxfun),
      calc.derivs = FALSE
    )
    fit <- glmer(formula, data = df, family = binomial, control = ctrl)
  } else {
    ctrl <- lmerControl(
      optimizer = "bobyqa",
      optCtrl = list(maxfun = maxfun),
      calc.derivs = FALSE
    )
    fit <- lmer(formula, data = df, REML = FALSE, control = ctrl)
  }
  fit
}

fit_status <- function(fit) {
  messages <- fit@optinfo$conv$lme4$messages
  opt_code <- fit@optinfo$conv$opt
  if (is.null(opt_code)) {
    opt_code <- 0
  }
  converged <- identical(as.numeric(opt_code), 0) && length(messages) == 0
  list(
    converged = converged,
    singular = isSingular(fit, tol = 1e-4),
    message = if (length(messages) == 0) "" else paste(messages, collapse = " | ")
  )
}

# A fit is "clean" when it converged with no non-singular warnings and is not
# singular; this drives reduction at the maximal/by-participant tiers. A
# "boundary" singular fit at the RI-only floor is still usable for the LRT.
conv_failed <- function(fit) {
  if (inherits(fit, "try-error")) return(TRUE)
  opt_code <- fit@optinfo$conv$opt
  if (is.null(opt_code)) opt_code <- 0
  if (!identical(as.numeric(opt_code), 0)) return(TRUE)
  msgs <- fit@optinfo$conv$lme4$messages
  if (length(msgs) == 0) return(FALSE)
  non_singular <- msgs[!grepl("singular|boundary", msgs, ignore.case = TRUE)]
  length(non_singular) > 0
}
fit_clean <- function(fit) {
  !conv_failed(fit) && !(!inherits(fit, "try-error") && isSingular(fit, tol = 1e-4))
}

# Fit base and predictor models under a shared random structure, reducing the
# structure together (maximal -> by-participant slope -> RI-only) until both fit
# cleanly; at the RI-only floor a singular boundary fit is retained. Returns the
# fitted pair, the selected tier, and convergence/singularity flags, or NULL if
# even the floor fails to converge.
fit_pair_tiered <- function(work, outcome_col, fixed_terms, predictor, family_name) {
  tiers <- c("correlated_maximal", "maximal", "by_participant", "ri")
  last <- NULL
  for (tier in tiers) {
    # base and pred share the SAME random structure (slope for the focal predictor);
    # they differ only in whether the focal predictor enters the fixed effects.
    base_formula <- make_formula(work, outcome_col, fixed_terms, re_struct = tier, slope_var = predictor)
    pred_formula <- make_formula(work, outcome_col, fixed_terms, predictor, re_struct = tier, slope_var = predictor)
    base_fit <- tryCatch(fit_one_model(base_formula, work, family_name), error = function(e) e)
    pred_fit <- tryCatch(fit_one_model(pred_formula, work, family_name), error = function(e) e)
    if (inherits(base_fit, "error") || inherits(pred_fit, "error")) {
      last <- list(tier = tier, error = TRUE,
                   message = conditionMessage(if (inherits(base_fit, "error")) base_fit else pred_fit))
      next
    }
    if (tier == "ri") {
      if (!conv_failed(base_fit) && !conv_failed(pred_fit)) {
        return(list(base = base_fit, pred = pred_fit, tier = tier,
                    base_status = fit_status(base_fit), pred_status = fit_status(pred_fit)))
      }
      last <- list(tier = tier, error = TRUE, message = "RI floor failed to converge")
    } else if (fit_clean(base_fit) && fit_clean(pred_fit)) {
      return(list(base = base_fit, pred = pred_fit, tier = tier,
                  base_status = fit_status(base_fit), pred_status = fit_status(pred_fit)))
    }
  }
  last
}

fit_pair <- function(df, outcome_part, outcome_col, family_name, level_name, group_name, predictor) {
  required <- unique(c(controls, predictor, outcome_col, "translator_id", "sentence_item_id", "task_type", "lang_pair"))
  work <- df[complete.cases(df[, required, drop = FALSE]), , drop = FALSE]
  if (nrow(work) < 30 || length(unique(work$translator_id)) < 2 || length(unique(work$sentence_item_id)) < 2) {
    return(data.frame(
      outcome_part = outcome_part,
      group = group_name,
      side = "tgt",
      label = outcome_col,
      level = level_name,
      predictor = paste0("control+", predictor),
      feature = predictor,
      metric = NA_real_,
      p_raw = NA_real_,
      n_participants = length(unique(work$translator_id)),
      n_items = length(unique(work$sentence_item_id)),
      n_rows = nrow(work),
      llf_control = NA_real_,
      llf_predictor = NA_real_,
      lr_stat = NA_real_,
      df_lrt = NA_real_,
      tier = NA_character_,
      base_converged = FALSE,
      predictor_converged = FALSE,
      base_singular = NA,
      predictor_singular = NA,
      status = "insufficient",
      message = "",
      stringsAsFactors = FALSE
    ))
  }

  fitted <- fit_pair_tiered(work, outcome_col, controls, predictor, family_name)
  if (is.null(fitted) || isTRUE(fitted$error)) {
    return(data.frame(
      outcome_part = outcome_part,
      group = group_name,
      side = "tgt",
      label = outcome_col,
      level = level_name,
      predictor = paste0("control+", predictor),
      feature = predictor,
      metric = NA_real_,
      p_raw = NA_real_,
      n_participants = length(unique(work$translator_id)),
      n_items = length(unique(work$sentence_item_id)),
      n_rows = nrow(work),
      llf_control = NA_real_,
      llf_predictor = NA_real_,
      lr_stat = NA_real_,
      df_lrt = NA_real_,
      tier = if (is.null(fitted)) NA_character_ else fitted$tier,
      base_converged = FALSE,
      predictor_converged = FALSE,
      base_singular = NA,
      predictor_singular = NA,
      status = "nonconverged",
      message = if (is.null(fitted)) "" else substr(as.character(fitted$message), 1, 500),
      stringsAsFactors = FALSE
    ))
  }

  fit_result <- tryCatch({
    base_fit <- fitted$base
    pred_fit <- fitted$pred
    base_status <- fitted$base_status
    pred_status <- fitted$pred_status
    ll0 <- as.numeric(logLik(base_fit))
    ll1 <- as.numeric(logLik(pred_fit))
    df0 <- attr(logLik(base_fit), "df")
    df1 <- attr(logLik(pred_fit), "df")
    df_lrt <- max(1, df1 - df0)
    lr_stat <- max(0, 2 * (ll1 - ll0))
    p_raw <- pchisq(lr_stat, df = df_lrt, lower.tail = FALSE)
    data.frame(
      outcome_part = outcome_part,
      group = group_name,
      side = "tgt",
      label = outcome_col,
      level = level_name,
      predictor = paste0("control+", predictor),
      feature = predictor,
      metric = (ll1 - ll0) / nobs(pred_fit),
      p_raw = p_raw,
      n_participants = length(unique(work$translator_id)),
      n_items = length(unique(work$sentence_item_id)),
      n_rows = nobs(pred_fit),
      llf_control = ll0,
      llf_predictor = ll1,
      lr_stat = lr_stat,
      df_lrt = df_lrt,
      tier = fitted$tier,
      base_converged = base_status$converged,
      predictor_converged = pred_status$converged,
      base_singular = base_status$singular,
      predictor_singular = pred_status$singular,
      status = "ok",
      message = paste(c(base_status$message, pred_status$message), collapse = " || "),
      stringsAsFactors = FALSE
    )
  }, error = function(e) {
    data.frame(
      outcome_part = outcome_part,
      group = group_name,
      side = "tgt",
      label = outcome_col,
      level = level_name,
      predictor = paste0("control+", predictor),
      feature = predictor,
      metric = NA_real_,
      p_raw = NA_real_,
      n_participants = length(unique(work$translator_id)),
      n_items = length(unique(work$sentence_item_id)),
      n_rows = nrow(work),
      llf_control = NA_real_,
      llf_predictor = NA_real_,
      lr_stat = NA_real_,
      df_lrt = NA_real_,
      tier = if (!is.null(fitted)) fitted$tier else NA_character_,
      base_converged = FALSE,
      predictor_converged = FALSE,
      base_singular = NA,
      predictor_singular = NA,
      status = paste0("failed:", class(e)[[1]]),
      message = substr(conditionMessage(e), 1, 500),
      stringsAsFactors = FALSE
    )
  })
  fit_result
}

data <- read.delim(input_path, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)
data$task_type <- factor(data$task_type)
data$lang_pair <- factor(data$lang_pair)
data$translator_id <- factor(data$translator_id)
data$sentence_item_id <- factor(data$sentence_item_id)

rows <- list()
idx <- 1
for (group_name in groups) {
  cat(sprintf("[two-part] group=%s\n", group_name))
  group_data <- filter_group(data, group_name)
  for (level_name in levels_to_run) {
    level_data <- group_data[group_data$level == level_name, , drop = FALSE]
    for (predictor in features) {
      rows[[idx]] <- fit_pair(
        level_data,
        "edit_probability",
        "edited",
        "binomial",
        level_name,
        group_name,
        predictor
      )
      idx <- idx + 1
      rows[[idx]] <- fit_pair(
        level_data,
        "positive_duration",
        "log_Dur_positive",
        "gaussian",
        level_name,
        group_name,
        predictor
      )
      idx <- idx + 1
    }
  }
}

out <- do.call(rbind, rows)
ok <- is.finite(out$p_raw)
out$q_bh <- NA_real_
out$q_bh[ok] <- p.adjust(out$p_raw[ok], method = "BH")
out$sig_bh <- is.finite(out$q_bh) & out$q_bh < 0.05

write.table(out, file = output_path, sep = "\t", quote = FALSE, row.names = FALSE)
cat(sprintf("[two-part] wrote %s\n", output_path))
