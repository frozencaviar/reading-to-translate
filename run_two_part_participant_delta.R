#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(lme4))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: <prepared_tsv> <output_tsv> [maxfun]")
input_path <- args[[1]]
output_path <- args[[2]]
maxfun <- if (length(args) >= 3) as.numeric(args[[3]]) else 100000

features <- c("tgt_surp", "xent", "eos_xatt", "y2srcseq_ent", "y2y", "y2cont", "y_ent")
controls <- c("tgt_seg_len", "tgt_freq", "tgt_posq")

fit_one <- function(formula, df, fam) {
  if (fam == "binomial") glmer(formula, data=df, family=binomial, control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=maxfun)))
  else lmer(formula, data=df, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=maxfun)))
}

make_formula <- function(df, outcome, pred = NULL) {
  rhs <- c(controls, pred)
  re <- "(1 | translator_id)"
  if (length(unique(df$sentence_item_id)) > 1) re <- paste(re, "(1 | sentence_item_id)", sep=" + ")
  as.formula(paste(outcome, "~", paste(c(rhs, re), collapse=" + ")))
}

obs_ll <- function(model, df) {
  X <- model.matrix(model)
  fe <- fixef(model)
  fm <- as.numeric(X %*% fe)
  re <- ranef(model)
  if ("translator_id" %in% names(re)) fm <- fm + re[["translator_id"]][as.character(df$translator_id), 1]
  if ("sentence_item_id" %in% names(re)) {
    item_re <- re[["sentence_item_id"]][as.character(df$sentence_item_id), 1]
    item_re[is.na(item_re)] <- 0
    fm <- fm + item_re
  }
  if (inherits(model, "glmerMod")) {
    phat <- fitted(model)
    eps <- 1e-15; phat <- pmax(eps, pmin(1-eps, phat))
    return(df$y * log(phat) + (1-df$y) * log(1-phat))
  }
  s <- sigma(model); if(is.na(s)||s<=0) s <- 1
  r <- df$y - fm
  -0.5*log(2*pi*s^2) - 0.5*(r/s)^2
}

df <- read.csv(input_path, sep="\t", stringsAsFactors=FALSE)
res <- list()

for (part in c("edit_probability", "positive_duration")) {
  outc <- if(part=="edit_probability") "edited" else "log_Dur_positive"
  fam <- if(part=="edit_probability") "binomial" else "gaussian"
  for (lv in c("token", "segment")) {
    w <- df[df$level == lv, ]
    # Only 'all' group
    w <- w[complete.cases(w[, c(controls, outc, "translator_id","sentence_item_id")]), ]
    if(nrow(w) < 50 || length(unique(w$translator_id)) < 5) next
    for (feat in features) {
      base_f <- make_formula(w, outc)
      pred_f <- make_formula(w, outc, feat)
      bf <- tryCatch(fit_one(base_f, w, fam), error=function(e) NULL)
      pf <- tryCatch(fit_one(pred_f, w, fam), error=function(e) NULL)
      if(is.null(bf) || is.null(pf)) next
      w$y <- w[[outc]]
      bl <- obs_ll(bf, w); pl <- obs_ll(pf, w)
      dl <- pl - bl
      ids <- w$translator_id
      for(pid in unique(ids)) {
        m <- ids == pid
        if(sum(m) < 5) next
        d <- mean(dl[m], na.rm=TRUE)
        if(is.finite(d)) res[[length(res)+1]] <- data.frame(
          outcome_part=part, group="all", side="tgt", label=outc, level=lv,
          feature=feat, translator_id=pid, delta_ll_per_obs=d, stringsAsFactors=FALSE)
      }
    }
  }
  cat(sprintf("  %s done\n", part))
}

rdf <- do.call(rbind, res)
write.table(rdf, output_path, sep="\t", row.names=FALSE, quote=FALSE)
cat(sprintf("Wrote %d rows\n", nrow(rdf)))
