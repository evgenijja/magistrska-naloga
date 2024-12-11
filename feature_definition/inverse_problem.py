import matlab.engine
import numpy as np

import os
import json

import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable


# contractility = np.arange(0.55, 2.2, 0.05)
# resistance = np.arange(0.7, 1.55, 0.05)
# compliance = np.arange(0.1, 1.35, 0.05)

def generate_wave():

    contractility = random.uniform(0.55, 2.2)
    resistance = random.uniform(0.7, 1.3)
    compliance = random.uniform(0.1, 1.3)

    p1, p2, p3 = contractility, resistance, compliance

    print(f"Generating wave with parameters: {p1} for contractility, {p2} for resistance and {p3} for compliance. ")

    counter = 999

    eng = matlab.engine.start_matlab()

    part, flow1, flow2, flow3, flow4, flow5, flow6, flow7, flow8, cav1, cav2, cav3, cav4, cav5, cav6, cav7, cav8 = eng.eval(f"CircAdapt4CompilerCall({p1}, {p2}, {p3}, {counter})", nargout=17)
    part, flow1, flow2, flow3, flow4, flow5, flow6, flow7, flow8 = list(np.array(part).flatten().tolist()), list(np.array(flow1).flatten().tolist()), list(np.array(flow2).flatten().tolist()), list(np.array(flow3).flatten().tolist()), list(np.array(flow4).flatten().tolist()), list(np.array(flow5).flatten().tolist()), list(np.array(flow6).flatten().tolist()), list(np.array(flow7).flatten().tolist()), list(np.array(flow8).flatten().tolist())
    # print(part)
    part = list(map(lambda x: x if type(x) != complex else -1, part))

    eng.quit()

    return part

def compute_reference_part(reference_filepath):

    with open(reference_filepath, 'r') as json_file:
        reference_data = json.load(json_file)

    reference_part = reference_data['1']["part"]
    return reference_part

def area_under_curve(part):
    area = 0.0
    n = len(part)

    for i in range(1, n):
        # Trapezoidal rule formula
        area += np.min([part[i-1], part[i+1]]) + (np.max([part[i-1], part[i+1]]) - np.min([part[i-1], part[i+1]])) * 0.5

    return area

def sum_of_squared_disatances(part1, part2):
    pass

def compute_feature(feature_name, part, reference_part=None):
    if feature_name == 'MAP':
        feature_value = np.min(part) + 1/3*(np.max(part) - np.min(part))

    elif feature_name == 'SP':
        feature_value = np.max(part)

    elif feature_name == 'DP':
        feature_value = np.min(part)

    elif feature_name == 'PP':
        feature_value = np.max(part) - np.min(part)

    # najdaljša absolutna razdalja do reference curve
    elif feature_name == 'min_absolute_distance':
        distances = []
        for i in range(len(part)):
            distances.append(np.abs(part[i] - reference_part[i]))
        feature_value = np.min(distances)

    elif feature_name == 'max_absolute_distance':
        distances = []
        for i in range(len(part)):
            distances.append(np.abs(part[i] - reference_part[i]))
        feature_value = np.max(distances)

    elif feature_name == 'area_under_curve':
        feature_value = area_under_curve(part)

    elif feature_name == 'ssd':
        pass



    return feature_value

def compute_values_for_all_waves(filepath, feature_name='DP'):
    """
    
    """

    result_dict = dict()
    counter = 0

    with open(filepath, 'r') as json_file:
        data = json.load(json_file)

    for key in data:
        params = data[key]["params"]
        part = data[key]["part"]

        PP = np.max(part) - np.min(part)
        MAP = np.min(part) + 1/3*(np.max(part) - np.min(part))
        feature_value = compute_feature(feature_name, part)

        result_dict[counter] = {
            "params" : params,
            "PP" : PP,
            "MAP" : MAP,
            "feature_value" : feature_value
        }
        counter += 1

    return result_dict

def plot_in_ranges(data, MAP_range, PP_range, feature_range, pt=None):

    X, Y, Z, C_val, C_diff = [], [], [], [], []

    X1, X2, X3 = [], [], []
    Y1, Y2, Y3 = [], [], []
    Z1, Z2, Z3 = [], [], []
    C1, C2, C3 = [], [], []
    C11, C22, C33 = [], [], []

    for key in data:
        coordinates = data[key]["params"]
        x, y, z = coordinates[0], coordinates[1], coordinates[2]
        MAP, PP, feature = data[key]["MAP"], data[key]["PP"], data[key]["feature_value"]

        MAP_condition = MAP <= MAP_range[1] and MAP >= MAP_range[0]
        PP_condition = PP <= PP_range[1] and PP >= PP_range[0]
        feature_condition = feature <= feature_range[1] and feature >= feature_range[0]

        if MAP_condition:
            X1.append(x)
            Y1.append(y)
            Z1.append(z)
            C1.append(MAP)

        if PP_condition:
            X2.append(x)
            Y2.append(y)
            Z2.append(z)
            C2.append(PP)

        if feature_condition:
            X3.append(x)
            Y3.append(y)
            Z3.append(z)
            C3.append(feature)

        if MAP_condition and PP_condition and feature_condition:
            X.append(x)
            Y.append(y)
            Z.append(z)
            

            # C11.append(MAP)
            # C22.append(PP)
            # C33.append(feature)

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f"Plot for pt: {(round(pt[0], 2), round(pt[1], 2), round(pt[2], 2))}")

    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(X1, Y1, Z1, c=C1, cmap='YlOrRd', marker='o')
    ax1.set_title('MAP')

    ax1.set_xlim(0.55, 2.2) # contractility
    ax1.set_ylim(0.7, 1.3) # resistance
    ax1.set_zlim(0.1, 1.3) # compliance

    ax1.scatter(pt[0], pt[1], pt[2], color = 'k', marker = 'o', s=50)

    # divider1 = make_axes_locatable(ax1)
    # cax1 = divider1.append_axes(size="5%", pad=0.05)
    # cbar1 = plt.colorbar(scatter1, cax=cax1, label='Z Values')

    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(X2, Y2, Z2, c=C2, cmap='Blues', marker='o')
    ax2.set_title('PP')

    ax2.set_xlim(0.55, 2.2) # contractility
    ax2.set_ylim(0.7, 1.3) # resistance
    ax2.set_zlim(0.1, 1.3) # compliance

    ax2.scatter(pt[0], pt[1], pt[2], color = 'k', marker = 'o', s=50)

    # divider2 = make_axes_locatable(ax2)
    # cax2 = divider2.append_axes(size="5%", pad=0.05)
    # cbar2 = plt.colorbar(scatter2, cax=cax2, label='Z Values')

    # ax3 = fig.add_subplot(233, projection='3d')
    # scatter3 = ax3.scatter(X3, Y3, Z3, c=C3, cmap='PuRd', marker='o')
    # ax3.set_title('FEATURE')

    # ax3.set_xlim(0.55, 2.2) # contractility
    # ax3.set_ylim(0.7, 1.3) # resistance
    # ax3.set_zlim(0.1, 1.3) # compliance

    # divider3 = make_axes_locatable(ax3)
    # cax3 = divider3.append_axes(size="5%", pad=0.05)
    # cbar3 = plt.colorbar(scatter3, cax=cax3, label='Z Values')

    # ax4 = fig.add_subplot(212, projection='3d')
    ax4 = fig.add_subplot(133, projection='3d')
    scatter4 = ax4.scatter(X, Y, Z, c='b', marker='o')
    ax4.set_title('INTERSECTION')

    ax4.set_xlim(0.55, 2.2) # contractility
    ax4.set_ylim(0.7, 1.3) # resistance
    ax4.set_zlim(0.1, 1.3) # compliance

    ax4.scatter(pt[0], pt[1], pt[2], color = 'k', marker = 'o', s=50)

    # cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
    # fig.colorbar(scatter4, cax=cbar_ax, label='Z Values')
    
    plt.tight_layout()

    # Show the plots
    plt.show()




if __name__ == "__main__":

    # choose the feature to focus on (besides MAP and PP)
    feature_name = 'DP'

    # generate random wave inside our ranges
    # part = generate_wave()


    # filepath = "results_20240226/results_test.json"
    # with open(filepath, 'r') as file:
    #     data = json.load(file)
    # part = data['5']['part']
    # params = data['5']['params']
    # print(params)

    opt1, opt2, opt3 = [0.9, 1.35, 1.85], [0.8, 1, 1.2], [0.3, 0.7, 1.1]
    pts = []
    for elt1 in opt1:
        for elt2 in opt2:
            for elt3 in opt3:
                pts.append([elt1, elt2, elt3])
    print(len(pts))

    filepath = "results_20240226/results.json"
    with open(filepath, 'r') as file:
        data = json.load(file)

    res = {}
    counter = 1
    for key in data:
        params = data[key]["params"]
        params = [round(params[0], 2), round(params[1], 2), round(params[2], 2)]
        # print(params)
        if params in pts:
            print(params)
            res[counter] = {"part" : data[key]["part"], "params" : data[key]["params"]}
            counter += 1


    pt = res[7]["params"]
    print(pt)
    part = res[7]["part"]

    # compute its MAP and PP and chosen feature
    PP = np.max(part) - np.min(part)
    MAP = np.min(part) + 1/3*(np.max(part) - np.min(part))
    feature_value = compute_feature(feature_name, part)

    print(f"Generated wave has MAP {MAP} and PP {PP} and feature {feature_value}")

    # define ranges
    MAP_range = [MAP - 2, MAP + 2]
    PP_range = [PP - 2, PP + 2]
    feature_range = [feature_value - 2, feature_value + 2]

    print(f"We are searching in ranges: {MAP_range} for MAP, {PP_range} for PP and {feature_range} for feature.")

    # compute values for all simulated waves
    filepath = "results_20240226/results.json"
    data = compute_values_for_all_waves(filepath, feature_name='DP')

    # plot in 3D the dots that correspond to similar MAP, PP and chosen feature
    plot_in_ranges(data, MAP_range, PP_range, feature_range, pt)
