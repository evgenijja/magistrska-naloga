df = read.csv("pca_data.csv", header=TRUE, sep=",")

res = prcomp(df)

prop_explained <- res$sdev^2 / sum(res$sdev^2)

new_df = data.frame(prop_explained, cumsum(prop_explained))

round(res$rotation,2)

biplot(res, cex=0.3)
