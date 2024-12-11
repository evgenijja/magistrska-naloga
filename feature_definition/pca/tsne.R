library(Rtsne)
library(ggplot2)

set.seed(321)


# TOČKE
df_tocke <- read.csv("normalized_pca_data_FINAL.csv", header = TRUE, sep = ",")
params <- df_tocke[,c(2:4)]
df_tocke <- df_tocke[, -c(1:4)]
df_tocke <- df_tocke[!duplicated(df_tocke),]

res_tocke <- Rtsne(df_tocke)
plot(res_tocke$Y, xlab = "t-SNE1", ylab = "t-SNE2")

y <- res_tocke$Y
write.csv(y, file = "tsne_tocke.csv", row.names = FALSE)
y_params <- cbind(y, params)
write.csv(y_params, file = "tsne_tocke_with_params.csv", row.names = FALSE)


# ZNAČILKE
df_znacilke <- read.csv("features_FINAL.csv", header = TRUE, sep = ',')
params <- df_znacilke[,c(2:4)]
df_znacilke <- df_znacilke[, -c(1:4)]
df_znacilke <- scale(df_znacilke)
length(df_znacilke[,1])
df_znacilke <- df_znacilke[!duplicated(df_znacilke),]
length(df_znacilke[,1])

res_znacilke <- Rtsne(df_znacilke)
plot(res_znacilke$Y, xlab = "t-SNE1", ylab = "t-SNE2")

y <- res_znacilke$Y
write.csv(y, file = "tsne_znacilke.csv", row.names = FALSE)
y_params <- cbind(y, params)
write.csv(y_params, file = "tsne_znacilke_with_params.csv", row.names = FALSE)
