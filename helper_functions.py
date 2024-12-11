import numpy as np

def area_under_curve(part):
    area = 0.0
    n = len(part)

    for i in range(1, n):
        # Trapezoidal rule formula
        area += np.min([part[i-1], part[i]]) + (np.max([part[i-1], part[i]]) - np.min([part[i-1], part[i]])) * 0.5

    return area