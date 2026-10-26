# 111_mi_sensitivity.R — multiple-imputation sensitivity analysis (mice, m = 20)
# for the primary adjusted association, versus median imputation with indicators
library(mice)
d <- read.csv('<PROJECT_ROOT>/exec/data/v6_mi_dataset.csv')
d$vent_24h <- as.integer(d$vent_24h)
d$aki_stage_24h <- as.integer(d$aki_stage_24h)
vars <- c('mort_28d', 'sicm', 'age', 'sofa_24h', 'cci', 'lactate_max_24h', 'vent_24h', 'aki_stage_24h')
dd <- d[, vars]

imp <- mice(dd, m = 20, method = 'pmm', seed = 20260101, printFlag = FALSE)
fit <- with(imp, glm(mort_28d ~ sicm + age + sofa_24h + cci + lactate_max_24h +
                       vent_24h + aki_stage_24h, family = binomial))
pool <- pool(fit)
s <- summary(pool, conf.int = TRUE, exponentiate = TRUE)
cat('=== B12 MI (m=20, pmm) pooled OR ===\n')
print(s[s$term == 'sicm', ])
or <- s[s$term == 'sicm', 'estimate']; lo <- s[s$term == 'sicm', 'X2.5..']; hi <- s[s$term == 'sicm', 'X97.5..']
cat(sprintf('MI adjusted OR = %.3f (%.3f-%.3f)\n', or, lo, hi))

# 对比: 中位数填补(现方法)
dd2 <- dd
dd2$lactate_max_24h[is.na(dd2$lactate_max_24h)] <- median(dd2$lactate_max_24h, na.rm = TRUE)
m2 <- glm(mort_28d ~ sicm + age + sofa_24h + cci + lactate_max_24h + vent_24h + aki_stage_24h,
          data = dd2, family = binomial)
ci <- exp(confint(m2))
cat(sprintf('Median-imputation adjusted OR = %.3f (%.3f-%.3f)\n', exp(coef(m2)['sicm']), ci['sicm', 1], ci['sicm', 2]))

out <- data.frame(method = c('MI_m20_pmm', 'median_imputation'),
                  OR = c(or, exp(coef(m2)['sicm'])),
                  lo = c(lo, ci['sicm', 1]), hi = c(hi, ci['sicm', 2]))
write.csv(out, '<PROJECT_ROOT>/exec/results/v6_mi_sensitivity.csv', row.names = FALSE)
cat('[done 111]\n')
