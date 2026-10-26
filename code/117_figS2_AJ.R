# 117_figS2_AJ.R — C4: 28天SICM累积发病率曲线 (Aalen-Johansen vs 朴素KM) + 死亡竞争事件
library(cmprsk)
d <- read.csv('<PROJECT_ROOT>/exec/data/comprisk_corrected.csv')
d <- d[!is.na(d$time) & !is.na(d$status), ]
d$status <- as.integer(d$status)
ci <- cuminc(d$time, d$status)
s1 <- ci$`1 1`; s2 <- ci$`1 2`
# 朴素KM (把死亡当删失)
km <- survival::survfit(survival::Surv(time, status == 1) ~ 1, data = d)

png('<PROJECT_ROOT>/exec/figures/figS2_cuminc_aj_km.png', width = 1600, height = 1300, res = 200)
par(mar = c(4.5, 4.5, 2.5, 1))
plot(s1$time, s1$est, type = 's', lwd = 2, col = '#d62728',
     xlab = 'Time from ICU admission (h)', ylab = 'Cumulative incidence',
     ylim = c(0, 0.35), xlim = c(0, 672),
     main = 'Cumulative incidence in the MIMIC new-onset cohort\n(n = 789; 215 SICM; death as competing event)')
lines(km$time, 1 - km$surv, type = 's', lwd = 1.6, col = '#7f7f7f', lty = 2)
lines(s2$time, s2$est, type = 's', lwd = 1.6, col = '#1f77b4')
legend('topleft', bty = 'n', lwd = 2, lty = c(1, 2, 1), col = c('#d62728', '#7f7f7f', '#1f77b4'),
       legend = c('SICM, Aalen-Johansen (death as competing event)',
                  'SICM, naive 1-Kaplan-Meier',
                  'Death before SICM (competing event)'))
abline(h = 0.246, col = '#d62728', lty = 3, lwd = 0.8)
text(330, 0.265, 'AJ 28-day estimate: 24.6%', col = '#d62728', cex = 0.9)
dev.off()
cat('[done 117: figS2 saved]\n')
