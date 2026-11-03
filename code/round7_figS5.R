suppressMessages(library(cmprsk))
d <- read.csv("exec/data/round7_comprisk_status.csv")
d$status <- as.integer(d$status)
ci <- cuminc(d$time,d$status)
s1 <- ci$`1 1`;s2 <- ci$`1 2`
km <- survival::survfit(survival::Surv(time,as.integer(status==1))~1,data=d)
png("exec/figures/figS2_cuminc_aj_km.png",width=2400,height=1950,res=300)
par(mar=c(4.5,4.5,2.5,2.5))
plot(s1$time,s1$est,type="s",lwd=2,col="#d62728",xlab="Time from ICU admission (h)",
 ylab="Cumulative incidence",ylim=c(0,0.55),xlim=c(0,672),
 main="Cumulative incidence in the MIMIC new-onset cohort\n(n = 613; 110 SICM events; death as competing event)")
lines(km$time,1-km$surv,type="s",lwd=1.6,col="#7f7f7f",lty=2)
lines(s2$time,s2$est,type="s",lwd=2,col="#1f77b4")
legend("topleft",bty="n",lwd=2,lty=c(1,2,1),col=c("#d62728","#7f7f7f","#1f77b4"),
 legend=c("SICM, Aalen-Johansen (death as competing event)","SICM, naive 1-Kaplan-Meier","Death before SICM (competing event)"))
abline(h=0.1794454,col="#d62728",lty=3,lwd=.8)
text(370,0.195,"AJ 28-day estimate: 17.9%",col="#d62728",cex=.9)
dev.off()
cat("[round7 FigureS5 saved]\n")
