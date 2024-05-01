import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline


if __name__=="__main__":
    df = pd.read_csv("feature_definition/pca/normalized_pca_results.csv")
    # df = pd.read_csv("feature_definition/pca/scaled_pca_results.csv")

    y1, y2 = df["PC1"], df["PC2"]
    x = list(np.arange(len(y1)))
    const = len(y1) * [0]

    fig, ax = plt.subplots(2, 1, figsize=(8, 12))
    ax[0].plot(x, y1)
    # draw constant line dashed and black
    ax[0].plot(x, const, 'k--')
    ax[1].plot(x, y2)
    ax[1].plot(x, const, 'k--')
    plt.tight_layout()
    plt.show()

    # dodaj na isti graf še cubic splines interpolacijo obeh komponent
