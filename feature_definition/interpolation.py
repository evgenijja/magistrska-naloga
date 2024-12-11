import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from reading_data import get_data, get_normal_data, extract_parts, normalize_parts, first_translation

from scipy.interpolate import CubicSpline


def cubic_splines_interpolation(x, y, plot=False):

    spl = CubicSpline(x, y)
    y_new = spl(x)

    # deriv_1 = spl(x, nu=1)
    # deriv_2 = spl(x, nu=2)
    
    x = np.array(x)

    # # Find minimum of the first derivative and maximum of the second derivative within [50, 200]
    # min_first_deriv = np.min(deriv_1[(x >= 50) & (x <= 200)])
    # max_second_deriv = np.max(deriv_2[(x >= 50) & (x <= 200)])

    if plot:
        fig, ax = plt.subplots(4, 1, figsize=(8, 12))
        ax[0].plot(x, spl(x))
        # ax[0].plot(x, y, 'o', label='data')
        ax[1].plot(x, spl(x, nu=1), '--', label='1st derivative')
        ax[2].plot(x, spl(x, nu=2), '--', label='2nd derivative')
        ax[3].plot(x, spl(x, nu=3), '--', label='3rd derivative')

        # ax[1].axvline(y=min_first_deriv, color='r', linestyle='--', label='Min 1st Derivative')
        # ax[2].axvline(y=max_second_deriv, color='b', linestyle='--', label='Max 2nd Derivative')
        x_interval = np.where((x >= 50) & (x <= 200))[0]
        min_1st_deriv_idx = np.argmin(spl(x[x_interval], nu=1))
        max_2nd_deriv_idx = np.argmax(spl(x[x_interval], nu=2))
        
        # Plot vertical dashed lines
        for j in range(4):
            ax[j].axvline(x[x_interval][min_1st_deriv_idx], color='r', linestyle='--', label='Min 1st deriv')
            ax[j].axvline(x[x_interval][max_2nd_deriv_idx], color='g', linestyle='--', label='Max 2nd deriv')
            ax[j].legend(loc='best')

        # for j in range(4):
        #     ax[j].legend(loc='best')
        plt.tight_layout()
        plt.show()

    return y_new




if __name__ == "__main__":

    y = get_normal_data(get_data())
    y = y[np.argmin(y):] + y[:np.argmin(y)]

    x = list(np.arange(len(y)))
    y_new = cubic_splines_interpolation(x, y, plot=True)





    
    