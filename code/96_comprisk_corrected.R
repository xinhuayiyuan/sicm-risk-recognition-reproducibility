# 96_comprisk_corrected.R - 修正队列竞争风险：CIF + Fine-Gray
library(cmprsk)
library(survival)
d <- read.csv("<PROJECT_ROOT>/exec/data/comprisk_corrected.csv")
d$status <- as.integer(d$status)
cat("n=", nrow(d), " status:", table(d$status), "\n")

ci <- cuminc(d$time, d$status)
tp <- timepoints(ci, times = c(72, 168, 672))$est
rn <- rownames(tp)
idx1 <- grep("(^| )1$", rn); idx2 <- grep("(^| )2$", rn)
km <- survfit(Surv(time, status == 1) ~ 1, data = d)
naive <- 1 - summary(km, times = c(72, 168, 672), extend = TRUE)$surv
res <- data.frame(time_h = c(72, 168, 672),
                  CIF_SICM = round(tp[idx1, ], 3),
                  naive_1KM = round(naive, 3),
                  CIF_death = round(tp[idx2, ], 3))
cat("\n=== corrected cohort CIF ===\n"); print(res)
write.csv(res, "<PROJECT_ROOT>/exec/results/v4_corrected_cif.csv", row.names = FALSE)

d$lact <- d$lactate_max_24h
d$lact[is.na(d$lact)] <- median(d$lact, na.rm = TRUE)
cov <- as.matrix(d[, c("lact", "sofa_24h", "age")])
fg <- crr(d$time, d$status, cov, failcode = 1, cencode = 0)
sfg <- summary(fg)
cat("\n=== Fine-Gray (corrected cohort) ===\n"); print(sfg)
fgtab <- data.frame(var = rownames(sfg$coef), HR = round(exp(sfg$coef[, 1]), 3),
                    p = signif(sfg$coef[, 5], 3))
write.csv(fgtab, "<PROJECT_ROOT>/exec/results/v4_corrected_finegray.csv", row.names = FALSE)
cat("\n[done]\n")
