# Run from project root. Patient-clustered Cox sensitivity models.
library(survival)
tv <- read.csv("exec/data/round7_time_varying.csv")
lm <- read.csv("exec/data/round7_landmark_mortality.csv")
lm$death_event <- as.integer(lm$death_event %in% c("True","TRUE","1",1))
extract <- function(fit, term, analysis, n, patients, events, deaths) {
  sm <- summary(fit)
  ci <- suppressMessages(confint(fit, level=0.95))
  data.frame(analysis=analysis,n=n,patients=patients,sicm_events=events,deaths=deaths,
             HR=unname(exp(coef(fit)[term])),HR_lo=unname(exp(ci[term,1])),
             HR_hi=unname(exp(ci[term,2])),p=sm$coefficients[term,"Pr(>|z|)"])
}
fit_tv <- coxph(Surv(start,stop,event) ~ sicm_td + age + gender_f + cci + cluster(subject_id),data=tv)
fit_lm <- coxph(Surv(duration,death_event) ~ early_sicm + age + gender_f + cci + cluster(subject_id),data=lm)
n_tv <- length(unique(tv$stay_id)); p_tv <- length(unique(tv$subject_id))
r_tv <- extract(fit_tv,"sicm_td","time_dependent_cox",n_tv,p_tv,
                sum(tapply(tv$sicm_td,tv$stay_id,max)),
                sum(tv$event))
r_lm <- extract(fit_lm,"early_sicm","landmark_24h_early_sicm",nrow(lm),length(unique(lm$subject_id)),
                sum(lm$early_sicm),sum(lm$death_event))
out <- rbind(r_tv,r_lm)
write.csv(out,"exec/results/round7_time_mortality.csv",row.names=FALSE)
print(summary(fit_tv)); print(summary(fit_lm)); print(out)
