# 62_comprisk.R - S5: 死亡为竞争事件的SICM累积发生率(CIF) + Fine-Gray
library(cmprsk)
library(survival)
d <- read.csv("<PROJECT_ROOT>/exec/data/comprisk_dn.csv")
cat("n=", nrow(d), " 事件分布:", table(d$status), "\n")

# 1. CIF vs 朴素累积发病率(1-KM把死亡当删失 -> 高估)
d$status <- as.integer(d$status)
ci <- cuminc(d$time, d$status)
tp <- timepoints(ci, times = c(72, 168, 672))$est
rn <- rownames(tp)
idx1 <- grep("(^| )1$", rn); idx2 <- grep("(^| )2$", rn)
cif_sicm_28 <- tp[idx1, ]
cif_death_28 <- tp[idx2, ]
# 朴素估计: 死亡删失后的1-KM
km <- survfit(Surv(time, status == 1) ~ 1, data = d)
km_est <- summary(km, times = c(72, 168, 672), extend = TRUE)$surv
naive <- 1 - km_est
res <- data.frame(time_h = c(72, 168, 672),
                  CIF_SICM_comprisk = round(cif_sicm_28, 3),
                  naive_1KM_SICM = round(naive, 3),
                  CIF_death = round(cif_death_28, 3))
cat("\n=== SICM累积发生率: 竞争风险CIF vs 朴素1-KM ===\n"); print(res)
write.csv(res, "<PROJECT_ROOT>/exec/results/v3_S5_cif.csv", row.names = FALSE)

# 2. Fine-Gray: SICM ~ lactate + sofa_24h + age
d$lact <- d$lactate_max_24h
d$lact[is.na(d$lact)] <- median(d$lact, na.rm = TRUE)
cov <- as.matrix(d[, c("lact", "sofa_24h", "age")])
fg <- crr(d$time, d$status, cov, failcode = 1, cencode = 0)
sfg <- summary(fg)
cat("\n=== Fine-Gray(结局=SICM, 竞争=死亡) ===\n"); print(sfg)
fgtab <- data.frame(var = rownames(sfg$coef), HR = round(exp(sfg$coef[, 1]), 3),
                    p = signif(sfg$coef[, 5], 3))
write.csv(fgtab, "<PROJECT_ROOT>/exec/results/v3_S5_finegray.csv", row.names = FALSE)
cat("\n[done]\n")
