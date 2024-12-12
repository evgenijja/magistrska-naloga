import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from scipy.interpolate import CubicSpline


import json

from reading_data import *

def transform_parts(list_of_lists, stable_part, mode=None):
    """
    Rišemo lahko različne valove:
    - navadne
    - normalizirane
    - ostanek (residum) ko odštejemo stabilen val

    stable_part je stabilen val pri 1, 1, 1 (nič ni transliran in normaliziran)
    """
    new_parts = []

    for elt in list_of_lists:

        
        part = elt['part']
        
        MAP = 1/3 * (np.max(part) - np.min(part)) + np.min(part)
        part = part[np.argmin(part):] + part[:np.argmin(part)]
        

        stable_part = stable_part[np.argmin(stable_part):] + stable_part[:np.argmin(stable_part)]

        if mode == "normalized":
            # normaliziramo
            pp = np.max(part) - np.min(part)
            dp = np.min(part)
            normalization_factor = 1 / pp 
            part = [(part[i]-dp) * normalization_factor for i in range(len(part))]

        elif mode == "residuum":
            part = [part[i] - stable_part[i] for i in range(len(part))]

        elif mode == "normalized_residuum":
            pp = np.max(part) - np.min(part)
            dp = np.min(part)
            normalization_factor = 1 / pp 
            part = [(part[i]-dp) * normalization_factor for i in range(len(part))]

            pp_stable = np.max(stable_part) - np.min(stable_part)
            dp_stable = np.min(stable_part)
            normalization_factor_stable = 1 / pp_stable
            stable_part = [(stable_part[i]-dp_stable) * normalization_factor_stable for i in range(len(stable_part))]

            part = [part[i] - stable_part[i] for i in range(len(part))]


        new_elt = {
            'C': elt['C'],
            'R': elt['R'],
            'S': elt['S'],
            'part': part,
            'MAP' : MAP
        }
        new_parts.append(new_elt)

    return new_parts

def parts_filter(list_of_lists, C_vals=[1], R_vals=[1], mode=None):
    """
    Risanje vseh valov je too much. Zato jih filtriramo.
    """
    new_list_of_lists = []

    for elt in list_of_lists:
        if mode == 'fixed C and R':
            if round(elt['C'], 2) in C_vals and round(elt['R'], 2) in R_vals:
                new_list_of_lists.append(elt)

    return new_list_of_lists




def plot_parts(list_of_lists, stable_part):
    """
    
    """

    stable_transformed_parts = transform_parts(stable_part, stable_part, mode=None)
    stable_transformed_part = stable_transformed_parts[0]['part']
    stable_normalized_parts = transform_parts(stable_part, stable_part, mode="normalized")
    stable_normalized_part = stable_normalized_parts[0]['part']

    stable_part = stable_part[0]['part']
    # stable_part = stable_part[np.argmin(stable_part):] #+ stable_part[:np.argmin(stable_part)]

    transformed_parts = transform_parts(list_of_lists, stable_part, mode=None)
    normalized_parts = transform_parts(list_of_lists, stable_part, mode="normalized")
    residuum_parts = transform_parts(list_of_lists, stable_part, mode="residuum")
    normalized_residuum_parts = transform_parts(list_of_lists, stable_part, mode="normalized_residuum")

    # definicija barve
    all_maps = [elt['MAP'] for elt in transformed_parts] + [elt['MAP'] for elt in stable_transformed_parts]
    colors = [(1, 1, 0), (1, 0, 0)]  # Yellow to red
    cmap_name = 'yellow_red'
    n_bins = len(all_maps)
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    color_indices = np.linspace(0, 1, n_bins)
    # print(color_indices)

    all_maps = sorted(all_maps)
    color_dict = dict((round(all_maps[i]), color_indices[i]) for i in range(len(all_maps)))
    print(color_dict)

    # x = np.arange(294)

    

    
    # color map bi morala napisat tko, da pogledaš min in max MAP value, potem pa razdeliš colormap na toliko delov kot je razlika in potem glede na to dodeliš barvo



    fig, axs = plt.subplots(3, 4, figsize=(14, 14))
    fig.suptitle(f"Comparison of different transformations of parts; MAP in [ {round(min(all_maps))}, {round(max(all_maps))}]")

    for elt in transformed_parts:
        x = np.arange(len(elt['part']))
        axs[0,0].plot(x, elt['part'], color=cmap(color_dict[round(elt['MAP'])]))
    axs[0,0].plot(x, stable_transformed_part, "k--")
    axs[0,0].set_title("Original waves")

    for elt in transformed_parts:
        spl = CubicSpline(x, elt['part'])
        # y_new = spl(x)
        axs[1,0].plot(x, spl(x, nu=1), color = cmap(color_dict[round(elt['MAP'])]))
    axs[1,0].set_title("1st derivative")
    spl = CubicSpline(x, stable_transformed_part)
    axs[1,0].plot(x, spl(x, nu=1), "k--")

    for elt in transformed_parts:
        spl = CubicSpline(x, elt['part'])
        # y_new = spl(x)
        axs[2,0].plot(x, spl(x, nu=2), color = cmap(color_dict[round(elt['MAP'])]))
    axs[2,0].set_title("2nd derivative")
    spl = CubicSpline(x, stable_transformed_part)
    axs[2,0].plot(x, spl(x, nu=2), "k--")
    
    for elt in normalized_parts:
        axs[0,1].plot(x, elt['part'], color=cmap(color_dict[round(elt['MAP'])]))
    axs[0,1].plot(x, stable_normalized_part, "k--")
    axs[0,1].set_title("Normalized waves")

    # 1st derivative
    for elt in normalized_parts:
        spl = CubicSpline(x, elt['part'])
        # y_new = spl(x)
        axs[1,1].plot(x, spl(x, nu=1), color = cmap(color_dict[round(elt['MAP'])]))
    axs[1,1].set_title("1st derivative")
    spl = CubicSpline(x, stable_normalized_part)
    axs[1,1].plot(x, spl(x, nu=1), "k--")

    # 2nd derivative
    for elt in normalized_parts:
        spl = CubicSpline(x, elt['part'])
        # y_new = spl(x)
        axs[2,1].plot(x, spl(x, nu=2), color = cmap(color_dict[round(elt['MAP'])]))
    axs[2,1].set_title("2nd derivative")
    spl = CubicSpline(x, stable_normalized_part)
    axs[2,1].plot(x, spl(x, nu=2), "k--")

    for elt in residuum_parts:
        axs[0,2].plot(x, elt['part'], color=cmap(color_dict[round(elt['MAP'])]))
    axs[0,2].set_title("Residuum waves")

    # 1st derivative
    for elt in residuum_parts:
        spl = CubicSpline(x, elt['part'])
        # y_new = spl(x)
        axs[1,2].plot(x, spl(x, nu=1), color = cmap(color_dict[round(elt['MAP'])]))
    axs[1,2].set_title("1st derivative")

    # 2nd derivative
    for elt in residuum_parts:
        spl = CubicSpline(x, elt['part'])
        # y_new = spl(x)
        axs[2,2].plot(x, spl(x, nu=2), color = cmap(color_dict[round(elt['MAP'])]))
    axs[2,2].set_title("2nd derivative")

    for elt in normalized_residuum_parts:
        axs[0,3].plot(x, elt['part'], color=cmap(color_dict[round(elt['MAP'])]))
    axs[0,3].set_title("Normalized residuum waves")

    # 1st derivative
    for elt in normalized_residuum_parts:
        spl = CubicSpline(x, elt['part'])
        # y_new = spl(x)
        axs[1,3].plot(x, spl(x, nu=1), color = cmap(color_dict[round(elt['MAP'])]))
    axs[1,3].set_title("1st derivative")

    # 2nd derivative
    for elt in normalized_residuum_parts:
        spl = CubicSpline(x, elt['part'])
        # y_new = spl(x)
        axs[2,3].plot(x, spl(x, nu=2), color = cmap(color_dict[round(elt['MAP'])]))
    axs[2,3].set_title("2nd derivative")

    plt.tight_layout()
    plt.show()


def plot_residuum(parts, stable):
    """
    residuum, 1st derivative, normalized residuum, 1st derivative normalized residuum
    """

    stable_transformed_parts = transform_parts(stable_part, stable_part, mode=None)
    stable_transformed_part = stable_transformed_parts[0]['part']
    stable_normalized_parts = transform_parts(stable_part, stable_part, mode="normalized")
    stable_normalized_part = stable_normalized_parts[0]['part']

    stable_part = stable_part[0]['part']

    transformed_parts = transform_parts(list_of_lists, stable_part, mode=None)
    normalized_parts = transform_parts(list_of_lists, stable_part, mode="normalized")
    residuum_parts = transform_parts(list_of_lists, stable_part, mode="residuum")
    normalized_residuum_parts = transform_parts(list_of_lists, stable_part, mode="normalized_residuum")

    # definicija barve
    all_maps = [elt['MAP'] for elt in transformed_parts] + [elt['MAP'] for elt in stable_transformed_parts]
    colors = [(1, 1, 0), (1, 0, 0)]  # Yellow to red
    cmap_name = 'yellow_red'
    n_bins = len(all_maps)
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    color_indices = np.linspace(0, 1, n_bins)
    # print(color_indices)

    all_maps = sorted(all_maps)
    color_dict = dict((round(all_maps[i]), color_indices[i]) for i in range(len(all_maps)))
    print(color_dict)

    x = np.arange(427)

    
    # color map bi morala napisat tko, da pogledaš min in max MAP value, potem pa razdeliš colormap na toliko delov kot je razlika in potem glede na to dodeliš barvo
    fig, axs = plt.subplots(4, 4, figsize=(14, 20))
    fig.suptitle(f"Comparison of different transformations of parts; MAP in [ {round(min(all_maps))}, {round(max(all_maps))}]")

    for elt in residuum_parts:
        spl = CubicSpline(x, elt['part'])
        # y_new = spl(x)
        axs[1,2].plot(x, spl(x, nu=1), color = cmap(color_dict[round(elt['MAP'])]))
    axs[1,2].set_title("1st derivative")

    # FILTRIRANJE!!!!

    # 2nd derivative
    for elt in residuum_parts:
        spl = CubicSpline(x, elt['part'])
        # y_new = spl(x)
        axs[2,2].plot(x, spl(x, nu=2), color = cmap(color_dict[round(elt['MAP'])]))
    axs[2,2].set_title("2nd derivative")

    for elt in normalized_residuum_parts:
        axs[0,3].plot(x, elt['part'], color=cmap(color_dict[round(elt['MAP'])]))
    axs[0,3].set_title("Normalized residuum waves")

    # 1st derivative
    for elt in normalized_residuum_parts:
        spl = CubicSpline(x, elt['part'])
        # y_new = spl(x)
        axs[1,3].plot(x, spl(x, nu=1), color = cmap(color_dict[round(elt['MAP'])]))
    axs[1,3].set_title("1st derivative")

    # 2nd derivative
    for elt in normalized_residuum_parts:
        spl = CubicSpline(x, elt['part'])
        # y_new = spl(x)
        axs[2,3].plot(x, spl(x, nu=2), color = cmap(color_dict[round(elt['MAP'])]))
    axs[2,3].set_title("2nd derivative")





if __name__ == "__main__":
    list_of_lists, stable_part = read_data()

    # contractility = np.arange(0.55, 2.2, 0.05)
    # resistance = np.arange(0.7, 1.55, 0.05)
    # compliance = np.arange(0.1, 1.35, 0.05)

    # C in R sta enaka 1, S je poljubno
    C_vals = [1]
    R_vals = [1]
    filtered_parts = parts_filter(list_of_lists, C_vals=C_vals, R_vals=R_vals, mode='fixed C and R')

    # # C in R sta nizka, S je poljubno
    # C_vals = [0.55]
    # R_vals = [0.7]
    # filtered_parts = parts_filter(list_of_lists, C_vals=C_vals, R_vals=R_vals, mode='fixed C and R')

    # # C in R sta visoka, S je poljubno
    # C_vals = [2.1]
    # R_vals = [1.3]
    # filtered_parts = parts_filter(list_of_lists, C_vals=C_vals, R_vals=R_vals, mode='fixed C and R')

    # # C je nizek, R je visok, S je poljubno
    # C_vals = [0.55]
    # R_vals = [1.3]
    # filtered_parts = parts_filter(list_of_lists, C_vals=C_vals, R_vals=R_vals, mode='fixed C and R')

    # # R je nizek, C je visok, S je poljubno
    # C_vals = [1.5, 1.7, 2.1]
    # R_vals = [0.7]
    # filtered_parts = parts_filter(list_of_lists, C_vals=C_vals, R_vals=R_vals, mode='fixed C and R')
    # # print(filtered_parts)

    new_filtered_parts = []

    for elt in filtered_parts:
        if len(set(elt['part'])) != 1:
            new_filtered_parts.append(elt)

    plot_parts(new_filtered_parts, stable_part)
    # new_parts = transform_parts(filtered_parts, stable_part, mode="normalized_residuum")

    
            

        

        



        

