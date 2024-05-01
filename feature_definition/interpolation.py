import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from reading_data import get_data, get_normal_data, extract_parts, normalize_parts, first_translation

from scipy.interpolate import CubicSpline


def cubic_splines_interpolation(x, y, plot=False):

    spl = CubicSpline(x, y)
    y_new = spl(x)

    if plot:
        fig, ax = plt.subplots(4, 1, figsize=(8, 12))
        ax[0].plot(x, spl(x))
        # ax[0].plot(x, y, 'o', label='data')
        ax[1].plot(x, spl(x, nu=1), '--', label='1st derivative')
        ax[2].plot(x, spl(x, nu=2), '--', label='2nd derivative')
        ax[3].plot(x, spl(x, nu=3), '--', label='3rd derivative')
        for j in range(4):
            ax[j].legend(loc='best')
        plt.tight_layout()
        plt.show()

    return y_new




if __name__ == "__main__":

    y = get_normal_data(get_data())
    y = y[np.argmin(y):] + y[:np.argmin(y)]

    x = list(np.arange(len(y)))
    y_new = cubic_splines_interpolation(x, y, plot=True)





    
    