import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json 

"""
Plot inverse plots.
"""

features3 = [
    "t_sys",
    "dicrotic_notch_idx",
    "razmerje2",
    "peak to dicrotic notch area",
    "min naklon",
    "max 2nd deriv after peak",
    "max normalized residuum",
    "min normalized residuum",
    "min 2nd derivative normalized residuum",
    "min 2nd derivative residuum"
    ]

features1 = [
    "PP",
    "SBP",
    "DBP",
    "rDNP",
    "t_sys",
    "dicrotic_notch_idx",
    "razmerje1",
    "razmerje2",
    "peak to dicrotic notch area",
    "dicrotic notch to end area",
    "max naklon",
    "min naklon",
    "max 2nd deriv after peak",
    "max normalized residuum",
    "min normalized residuum",
    "min 1st derivative normalized residuum",
    "max 1st derivative normalized residuum",
    "min 2nd derivative normalized residuum",
    "max 2nd derivative normalized residuum",
    "max residuum",
    "min residuum",
    "min 1st derivative residuum",
    "max 1st derivative residuum",
    "min 2nd derivative residuum",
    "max 2nd derivative residuum",
    ]

reference_pts = [
    # nobena
    [0.55 + 0.4, 0.7 + 0.15, 0.1 + 0.3],

    # vse
    [0.55 + 3 * 0.4, 0.7 + 3 * 0.15, 0.1 + 3 * 0.3],

    # po dve
    [0.55 + 0.4, 0.7 + 3 * 0.15, 0.1 + 3 * 0.3],
    [0.55 + 3 * 0.4, 0.7 + 3 * 0.15, 0.1 + 0.3],
    [0.55 + 3 * 0.4, 0.7 + 0.15, 0.1 + 3 * 0.3],
    
    # ena
    [0.55 + 3 * 0.4, 0.7 + 0.15, 0.1 + 0.3],
    [0.55 + 0.4, 0.7 + 3 * 0.15, 0.1 + 0.3],
    [0.55 + 0.4, 0.7 + 0.15, 0.1 + 3 * 0.3]
    ]

new_reference_pts = []
for pt in reference_pts:
    pt = [round(elt, 3) for elt in pt]
    new_reference_pts.append(pt)

reference_pts = new_reference_pts

# read data
def get_data():
    """
    Dobimo rezultate vseh simulacij. 
    """
    with open("data/results.json", "r") as json_file:
        data = json.load(json_file)

    # nekatere parametre ful čudno zaokroži
    for key in data:
        params = data[key]['params']
        new_params = [round(params[0], 3), round(params[1], 3), round(params[2], 3)]
        data[key]['params'] = new_params

    return data

def make_inverse_plot(reference_point, feature_name, confidence, data):
    """
    Identify the points that are in the confidence rango of the chosen feature
    Featurje smo že poračunali enkrat v features tabeli    
    """
    
    reference_feature = list(data[(data["C"] == reference_point[0]) & (data["R"] == reference_point[1]) & (data["S"] == reference_point[2])][feature_name])[0]
    # print(f"reference: {reference_feature}")
    lower_bound = reference_feature - confidence
    upper_bound = reference_feature + confidence

    
    # print(f"lower and upper: {(lower_bound, upper_bound)}")

    # get all points in the confidence interval
    points = data[(data[feature_name] >= lower_bound) & (data[feature_name] <= upper_bound)]
    # print(points.head())
    C, R, S = [elt for elt in points["C"]], [elt for elt in points["R"]], [elt for elt in points["S"]]
    points_to_plot = [[C[i], R[i], S[i]] for i in range(len(C))]
    # print(points_to_plot)

    x_coords = [point[0] for point in points_to_plot]
    y_coords = [point[1] for point in points_to_plot]
    z_coords = [point[2] for point in points_to_plot]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(x_coords, y_coords, z_coords, c='b', marker='o')

    # Plot the fixed point in black with a larger size
    ax.scatter(reference_point[0], reference_point[1], reference_point[2], c='k', marker='o', s=100)

    # Set labels
    ax.set_xlabel('C')
    ax.set_ylabel('R')
    ax.set_zlabel('S')

    # title
    ax.set_title(f"Feature: {feature_name}, confidence: {confidence}")

    # Show the plot
    # plt.show()

def make_joint_inverse_plot(reference_point, feature_name, data, confidence, ax, confidence_MAP=2, confidence_PP = 2):
    """
    Identify the points that are in the confidence rango of the chosen feature
    Featurje smo že poračunali enkrat v features tabeli    
    """
    # compute feature value and confidence values
    reference_feature = list(data[(data["C"] == reference_point[0]) & (data["R"] == reference_point[1]) & (data["S"] == reference_point[2])][feature_name])[0]
    lower_bound = reference_feature - confidence
    upper_bound = reference_feature + confidence

    # get all points in the confidence interval
    points = data[(data[feature_name] >= lower_bound) & (data[feature_name] <= upper_bound)]
    C, R, S = [elt for elt in points["C"]], [elt for elt in points["R"]], [elt for elt in points["S"]]
    points_to_plot = [[C[i], R[i], S[i]] for i in range(len(C))]

    # compute map value and confidence value
    reference_MAP = list(data[(data["C"] == reference_point[0]) & (data["R"] == reference_point[1]) & (data["S"] == reference_point[2])]["MAP"])[0]
    lower_bound_MAP = reference_MAP - confidence_MAP
    upper_bound_MAP = reference_MAP + confidence_MAP

    # get all points in the confidence interval
    points_MAP = data[(data["MAP"] >= lower_bound_MAP) & (data["MAP"] <= upper_bound_MAP)]
    C_MAP, R_MAP, S_MAP = [elt for elt in points_MAP["C"]], [elt for elt in points_MAP["R"]], [elt for elt in points_MAP["S"]]
    points_to_plot_MAP = [[C_MAP[i], R_MAP[i], S_MAP[i]] for i in range(len(C_MAP))]

    # compute PP value and confidence value
    reference_PP = list(data[(data["C"] == reference_point[0]) & (data["R"] == reference_point[1]) & (data["S"] == reference_point[2])]["PP"])[0]
    lower_bound_PP = reference_PP - confidence_PP
    upper_bound_PP = reference_PP + confidence_PP

    # get all points in the confidence interval
    points_PP = data[(data["PP"] >= lower_bound_PP) & (data["PP"] <= upper_bound_PP)]
    C_PP, R_PP, S_PP = [elt for elt in points_PP["C"]], [elt for elt in points_PP["R"]], [elt for elt in points_PP["S"]]
    points_to_plot_PP = [[C_PP[i], R_PP[i], S_PP[i]] for i in range(len(C_PP))]

    # save points that are in all three confidence intervals
    final_points_to_plot = [elt for elt in points_to_plot if elt in points_to_plot_MAP and elt in points_to_plot_PP]

    x_coords = [point[0] for point in final_points_to_plot]
    y_coords = [point[1] for point in final_points_to_plot]
    z_coords = [point[2] for point in final_points_to_plot]

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(x_coords, y_coords, z_coords, c='b', marker='o')

    # Plot the fixed point in black with a larger size
    ax.scatter(reference_point[0], reference_point[1], reference_point[2], c='k', marker='o', s=100)

    # Set labels
    ax.set_xlabel('C')
    ax.set_ylabel('R')
    ax.set_zlabel('S')

    # [0.55, 2.15] [0.7, 1.3] [0.1, 1.3]
    ax.set_xlim([0.55, 2.15])
    ax.set_ylim([0.7, 1.3])
    ax.set_zlim([0.1, 1.3])

    # title
    ax.set_title(f"Feature: {feature_name},\n confidence: {confidence},\n point of reference: {reference_point}")

    # Show the plot
    # plt.show()

def plot_all_reference_plots(reference_points, feature_name, data, confidence, confidence_MAP=2, confidence_PP = 2):

    fig, axes = plt.subplots(2, 4, subplot_kw={'projection': '3d'}, figsize=(15, 10))

# Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Plot using custom function with different data and titles
    for i in range(len(reference_points)):
        reference_point = reference_points[i]
        print(reference_point)
        make_joint_inverse_plot(reference_point, feature_name, data, confidence, ax=axes[i], confidence_MAP=2, confidence_PP = 2)

    plt.tight_layout()
    plt.show()


def find_intervals(data):

    C_interval = [min(data["C"]), max(data["C"])]
    R_interval = [min(data["R"]), max(data["R"])]
    S_interval = [min(data["S"]), max(data["S"])]

    return C_interval, R_interval, S_interval


if __name__ == "__main__":

    data = get_data()

    data = pd.read_csv("data/features.csv")

    # C_interval, R_interval, S_interval = find_intervals(data)
    # print(C_interval, R_interval, S_interval)
    # [0.55, 2.15] [0.7, 1.3] [0.1, 1.3]
    # koraki: 0.4, 0.15, 0.3

    

    # make_inverse_plot([1, 1, 1], "PP", 0.1, data)

    # generate random point from data

    # plot points in the confidence interval

    # for feature in features3:
    #     make_joint_inverse_plot([0.9, 0.9, 0.9], feature, data, 2)

    plot_all_reference_plots(reference_pts, "MAP", data, 2, confidence_MAP=2, confidence_PP = 2)

    

