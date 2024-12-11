# ("feature_definition/pca/normalized_pca_data_10092024_2.csv")

df <- read.csv("normalized_pca_data_FINAL.csv", header = TRUE, sep = ",")
df <- df[, -1]


# principal component analysis
res <- prcomp(df)

# get percentage of explained varience
prop_explained <- res$sdev^2 / sum(res$sdev^2)
new_df <- data.frame(prop_explained, cumsum(prop_explained))
write.csv(new_df, file = "pca_results2.csv", row.names = FALSE)

# loadings
loadings <- res$rotation
write.csv(loadings, "loadings_results2.csv", row.names = FALSE)

# scores
scores = res$x
write.csv(scores, "scores_results2.csv", row.names = FALSE)


# na značilkah
df <- read.csv("features_FINAL.csv", header = TRUE, sep = ',')
df <- df[, -c(1:4)]

df <- scale(df)

res <- prcomp(df)

# get percentage of explained varience
prop_explained <- res$sdev^2 / sum(res$sdev^2)
new_df <- data.frame(prop_explained, cumsum(prop_explained))
write.csv(new_df, file = "pca_results2_znacilke.csv", row.names = FALSE)

# loadings
loadings <- res$rotation
write.csv(loadings, "loadings_results2_znacilke.csv", row.names = FALSE)

# scores
scores = res$x
write.csv(scores, "scores_results2_znacilke.csv", row.names = FALSE)
