# Round-7 competing-risk incidence and Fine-Gray sensitivity.
library(cmprsk)
library(survival)
d <- read.csv("exec/data/round7_cohort_timing.csv")
d <- d[(d$has_hf==0)&(d$congestive_heart_failure==0),]
CENS <- 672
resolve <- function(ts,td) {
  if (is.na(td) || td<0) td <- NA_real_
  if (!is.na(ts) && (is.na(td) || ts<td)) return(c(min(ts,CENS),1))
  if (!is.na(td)) return(c(min(td,CENS),2))
  if (!is.na(ts)) return(c(min(ts,CENS),1))
  return(c(CENS,0))
}
d$time <- CENS; d$status <- 0L
for (i in seq_len(nrow(d))) {
  ts <- if (d$sicm[i]==1) d$sicm_onset_h[i] else NA_real_
  z <- resolve(ts,d$death_h[i]);d$time[i] <- z[1];d$status[i] <- as.integer(z[2])
}
ci <- cuminc(d$time,d$status)
tp <- timepoints(ci,times=c(72,168,672))$est
rn <- rownames(tp);i1 <- 1;i2 <- 2
km <- survfit(Surv(time,as.integer(status==1))~1,data=d)
naive <- 1-summary(km,times=c(72,168,672),extend=TRUE)$surv
print(table(d$status));print(tp);print(naive)
out <- data.frame(time_h=c(72,168,672),CIF_SICM=tp[i1,],naive_1KM=naive,CIF_death=tp[i2,])
write.csv(out,"exec/results/round7_comprisk_cif.csv",row.names=FALSE)
d$lact <- d$lactate_max_24h;d$lact[is.na(d$lact)] <- median(d$lact,na.rm=TRUE)
fg <- crr(d$time,d$status,as.matrix(d[,c("lact","sofa_24h","age")]),failcode=1,cencode=0)
sfg <- summary(fg)
fgtab <- data.frame(var=rownames(sfg$coef),HR=exp(sfg$coef[,1]),lo=sfg$conf.int[,3],hi=sfg$conf.int[,4],p=sfg$coef[,5])
write.csv(fgtab,"exec/results/round7_comprisk_finegray.csv",row.names=FALSE)
write.csv(d,"exec/data/round7_comprisk_status.csv",row.names=FALSE)
print(table(d$status));print(out);print(fgtab)
