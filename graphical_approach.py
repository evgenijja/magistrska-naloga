import numpy as np
import pandas as pd
import json

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from feature_renaming import renaming

blue_scale_r = {
    0.7: "#b3e4ff",   # Lighter than previous "#a0ddff"
    0.75: "#93d7ff",  # Lighter than previous "#80d0ff"
    0.8: "#73cbff",   # Lighter than previous "#60c3ff"
    0.85: "#53beff",  # Lighter than previous "#40b6ff"
    0.9: "#33b1ff",   # Lighter than previous "#20a9ff"
    0.95: "#13a5ff",  # Lighter than previous "#009cff"
    1: "#0096e6",     # Lighter than previous "#008de6"
    1.05: "#0080cc",  # Lighter than previous "#007fcc"
    1.1: "#0072b3",   # Lighter than previous "#0071b3"
    1.15: "#006599",  # Lighter than previous "#006399"
    1.2: "#005780",   # Lighter than previous "#005580"
    1.25: "#004966",  # Lighter than previous "#004766"
    1.3: "#003c4d",   # Lighter than previous "#00394d"
    1.35: "#002e33",  # Lighter than previous "#002b33"
    1.4: "#00221a",   # Lighter than previous "#001e1a"
    1.45: "#001533",  # Lighter than previous "#001033"
    1.5: "#00101a",   # Lighter than previous "#000b1a"
    1.55: "#000a1a"   # Lighter than previous "#00061a"
}

blue_scale_c = {
    0.5: "#c6edff",
    0.55: "#c0eaff",
    0.6: "#bae7ff",
    0.65: "#b3e4ff",  # matches original at 0.7
    0.7: "#b3e4ff",   # Lighter than previous
    0.75: "#93d7ff",  # matches original
    0.8: "#73cbff",   # matches original
    0.85: "#53beff",  # matches original
    0.9: "#33b1ff",   # matches original
    0.95: "#13a5ff",  # matches original
    1.0: "#0096e6",   # matches original
    1.05: "#0080cc",  # matches original
    1.1: "#0072b3",   # matches original
    1.15: "#006599",  # matches original
    1.2: "#005780",   # matches original
    1.25: "#004966",  # matches original
    1.3: "#003c4d",   # matches original
    1.35: "#002e33",  # matches original
    1.4: "#00221a",   # matches original
    1.45: "#001533",  # matches original
    1.5: "#00101a",   # matches original
    1.55: "#000a1a",  # matches original
    1.6: "#00081a",
    1.65: "#000619",
    1.7: "#000518",
    1.75: "#000416",
    1.8: "#000314",
    1.85: "#000213",
    1.9: "#000111",
    1.95: "#00000f",
    2.0: "#00000d",
    2.05: "#00000b",
    2.1: "#00000a",
    2.15: "#000008"
}

blue_scale_s = {
    0.1: "#e6f7ff",
    0.15: "#dff4ff",
    0.2: "#d9f1ff",
    0.25: "#d3eeff",
    0.3: "#cdeaff",
    0.35: "#c6e7ff",
    0.4: "#c0e4ff",
    0.45: "#bae0ff",
    0.5: "#b3ddff",   # Slightly darker than original 0.7
    0.55: "#add9ff",
    0.6: "#a7d5ff",
    0.65: "#a0d1ff",
    0.7: "#9acfff",   # Approximates 0.7 from original
    0.75: "#93d7ff",  # matches original
    0.8: "#73cbff",   # matches original
    0.85: "#53beff",  # matches original
    0.9: "#33b1ff",   # matches original
    0.95: "#13a5ff",  # matches original
    1.0: "#0096e6",   # matches original
    1.05: "#0080cc",  # matches original
    1.1: "#0072b3",   # matches original
    1.15: "#006599",  # matches original
    1.2: "#005780",   # matches original
    1.25: "#004966",  # matches original
    1.3: "#003c4d"    # matches original
}


def read_parsed_data():

    with open("data/parsed_data2.json", "r") as json_file:
        data = json.load(json_file)

    new_data, stable_part = [], []
    for point in data:
        part = point['part']
        if point['params'] == [1, 1, 1]:
            stable_part.append(point)
        if not np.isnan(part).any() and part != len(part) * [0] and len(set(part)) > 1 and sum([0 if elt >= 0 else 1 for elt in part]) == 0:
            if point not in new_data:
                new_data.append(point)

    return new_data, stable_part

def load_new_data (filepath="data/features_FINAL.xlsx"):

    # read dataframe that has and index column and header
    df = pd.read_excel(filepath, header=0, index_col=0)
    # df = df[:100]
    print(len(df))
    df.dropna(inplace=True)
    print(len(df))
    C, R, S = df['C'], df['R'], df['S']
    
    # X is all the columns that are not C, R, S
    feature_names = df.columns.tolist()
    feature_names.remove('C')
    feature_names.remove('R')
    feature_names.remove('S')
    X = df[feature_names]

    return X, C, R, S, feature_names, df

def plot_bp_vs_r(data, param="R", mode="dbp_vs_sbp", blue_scale=blue_scale_r):
    """
    Plot dbp vs sbp for all waves and their color should be the value of R
    """

    scale_values = np.array(list(blue_scale.keys()))
    colors = list(blue_scale.values())

    # Create a colormap from the custom scale
    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(scale_values, cmap.N)

    dbp, sbp = [], []
    map_values, pp_values = [], []
    cs, rs, ss = [], [], []
    for elt in data:
        part = elt['part']
        rs.append(elt['params'][1])
        cs.append(elt['params'][0])
        ss.append(elt['params'][2])

        dbp.append(np.min(part))
        sbp.append(np.max(part))

        map_values.append(1/3 * (np.max(part) - np.min(part)) + np.min(part))
        pp_values.append(np.max(part))

    # Scatter plot
    if param == "R":
        scatter = plt.scatter(dbp, sbp, c=rs, cmap=cmap, norm=norm, s=5)
    elif param == "C":
        scatter = plt.scatter(dbp, sbp, c=cs, cmap=cmap, norm=norm, s=5)
    elif param == "S":
        scatter = plt.scatter(dbp, sbp, c=ss, cmap=cmap, norm=norm, s=5)
    # scatter = plt.scatter(map_values, pp_values, c=rs, cmap=cmap, norm=norm, s=5)

    # Adding the colorbar
    cbar = plt.colorbar(scatter, ticks=scale_values)

    if param == "R":
        cbar.set_label('vrednost parametra za upornost')
    elif param == "C":
        cbar.set_label('vrednost parametra za kontraktilnost')
    elif param == "S":
        cbar.set_label('vrednost parametra za podajnost')
    # cbar.set_label('vrednost parametra za upornost')  # Label for the colorbar

    if mode == "dbp_vs_sbp":
        plt.xlabel("diastolni tlak")
        plt.ylabel("sistolni tlak")
    elif mode == "map_vs_pp":
        plt.xlabel("srednji krvni tlak")
        plt.ylabel("pulzni tlak")

    plt.savefig(f"final_plots/{mode}_color_{param}.png")
    # plt.savefig("final_plots/map_vs_pp_color_r.png")
    # plt.title("DBP vs SBP")
    # plt.show()
    plt.close()

def nearby_pts(df, feature_name, threshold=0.02, plot=False):

    """
    For each pt get t_sp
    """
    def count_close_values(row, col, threshold=0.02):
        return np.sum(np.abs(col - row) <= threshold)
    
    df["count"] = df[feature_name].apply(lambda x: count_close_values(x, df[feature_name]))
    
    if plot:
        x, y, z = df["C"], df["R"], df["S"]
        count = df['count']
        cmap = cm.get_cmap('Blues')
        norm = mcolors.Normalize(vmin=min(count), vmax=max(count))

        # Create the figure and 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot
        sc = ax.scatter(x, y, z, c=count, cmap=cmap, norm=norm)

        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('število sosednjih točk')

        # Labels and title
        ax.set_xlabel('kontraktilnost')
        ax.set_ylabel('upornost')
        ax.set_zlabel('podajnost')
        # ax.set_title('3D Scatter Plot with Color Representing Count')

        ax.set_xlim(0.55, 2.15)
        ax.set_ylim(0.7, 1.5)
        ax.set_zlim(0.1, 1.3)

        plt.savefig("final_plots/nearby_pts_t_sp.png")

        plt.show()


def plot_inverse_feature_with_confidence(data, feature_dict, ax=None):
    """
    Plot inverse plot of all waves that have feature value within confidence.
    """

    all_points_to_plot = []

    for feature in feature_dict:

        confidence = feature_dict[feature]

        lower_bound = confidence[0]
        upper_bound = confidence[1]

        # get all points in the confidence interval
        points = data[(data[feature] >= lower_bound) & (data[feature] <= upper_bound)]

        C, R, S = [elt for elt in points["C"]], [elt for elt in points["R"]], [elt for elt in points["S"]]
        points_to_plot = [(C[i], R[i], S[i]) for i in range(len(C))]
        # print(points_to_plot)

        all_points_to_plot.append(points_to_plot)

    intersection = set(all_points_to_plot[0])
    for i in range(1, len(all_points_to_plot)):
        intersection = intersection.intersection(set(all_points_to_plot[i]))

    intersection = list(intersection)

    print(len(intersection))

    x_coords = [point[0] for point in intersection]
    y_coords = [point[1] for point in intersection]
    z_coords = [point[2] for point in intersection]

    # shape
    # if np.max(x_coords) - np.min(x_coords) <= 0.4: # 1/4 of the range
    #     x_shape = "narrow"
    # else:
    #     x_shape = "wide"

    # if np.max(y_coords) - np.min(y_coords) <= 0.2: # 1/4 of the range
    #     y_shape = "narrow"
    # else:
    #     y_shape = "wide"

    # if np.max(z_coords) - np.min(z_coords) <= 0.3: # 1/4 of the range
    #     z_shape = "narrow"
    # else:
    #     z_shape = "wide"
    
    # shapes = [x_shape, y_shape, z_shape]
    # wides = shapes.count("wide")
    # narrows = shapes.count("narrow")

    # if wides == 3:
    #     shape = "prostor"
    # elif wides == 2 and narrows == 1:
    #     shape = "ploskev"
    # elif wides == 1 and narrows == 2:
    #     shape = "valj"
    # elif narrows == 3:
    #     shape = "točka"

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # plot one extra point in black and thicker
    
    if len(x_coords) >= 0:
        ax.scatter(x_coords, y_coords, z_coords)
        # ax.scatter(reference_pt["C"], reference_pt["R"], reference_pt["S"], color='k', s=15)

        ax.set_xlabel('kontraktilnost', fontdict={'fontsize': 15})
        ax.set_ylabel('upornost', fontdict={'fontsize': 15})
        ax.set_zlabel('podajnost', fontdict={'fontsize': 15})

        # ax.set_title(f"Število točk: {len(x_coords)}, Oblika: {shape} \n C: [{round(np.min(x_coords), 2)}, {round(np.max(x_coords), 2)}],\n Upornost: [{round(np.min(y_coords), 2)}, {round(np.max(y_coords), 2)}], \n Podajnost: [{round(np.min(z_coords), 2)}, {round(np.max(z_coords), 2)}]")
        title = ""
        for feature in feature_dict:
            confidence = feature_dict[feature]
            title += f"{feature}: {confidence[0]} - {confidence[1]},\n"

        # ax.set_title(f"Število točk: {len(x_coords)}, Oblika: {shape} \n C: [{round(np.min(x_coords), 2)}, {round(np.max(x_coords), 2)}],\n Upornost: [{round(np.min(y_coords), 2)}, {round(np.max(y_coords), 2)}], \n Podajnost: [{round(np.min(z_coords), 2)}, {round(np.max(z_coords), 2)}]")
        ax.set_title(title[:-2], fontdict={'fontsize': 15})
        # ranges: 
        # C : 0.55 - 2.15
        # R : 0.7 - 1.5
        # S :  0.1 - 1.3

        ax.set_xlim((0.55, 2.15))
        ax.set_ylim((0.7, 1.5))
        ax.set_zlim((0.1, 1.3))

        plt.tight_layout()
        if ax is None:
            plt.show()

    

def plot_multiple_plots(data, feature_dict):

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), subplot_kw={'projection': '3d'})

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # for reference_pt, ax in zip(reference_pts, axes):

    for i in range(len(feature_dict[list(feature_dict.keys())[0]])):
        f_dict = dict((feature, feature_dict[feature][i]) for feature in feature_dict)
        print(f_dict)
        plot_inverse_feature_with_confidence(data, f_dict, ax=axes[i])

    plt.tight_layout()
    # plt.savefig(filepath)
    plt.show()

def plot_waves(data, lower, upper):

    x = np.arange(0, len(data[0]['part']))
    parts_to_plot = []
    pps = []

    for elt in data:
        part = elt['part']
        map_value = 1/3 * (np.max(part) - np.min(part)) + np.min(part)

        pp = np.max(part) - np.min(part)
        pps.append(round(pp))

        if map_value >= lower and map_value <= upper:
            parts_to_plot.append(part)
            # plt.plot(part)
            # plt.show()

    # count how many times each pp value appears and print it as a dictiona
    pp_counter = dict((pp, pps.count(pp)) for pp in pps)
    # order the dictionary by the key ascending
    pp_counter = dict(sorted(pp_counter.items(), key=lambda item: item[0]))
    print(pp_counter)

    for part in parts_to_plot:
        plt.plot(x, part)
    plt.title(f"Število valov kjer MAP med {lower} in {upper}")
    plt.show()

def plot_map_progression_with_pp(data):

    fig, axes = plt.subplots(1, 5, figsize=(20, 10), subplot_kw={'projection': '3d'})

    axes = axes.flatten()



    # f_dict = {"MAP" : [70, 75]}
    # plot_inverse_feature_with_confidence(data, f_dict, ax=axes[0])
    # print("1")
    # f_dict = {"MAP" : [70, 75], "PP" : [25, 35]}
    # plot_inverse_feature_with_confidence(data, f_dict, ax=axes[1])
    # print("2")
    # f_dict = {"MAP" : [70, 75], "PP" : [35, 45]}
    # plot_inverse_feature_with_confidence(data, f_dict, ax=axes[2])
    # print("3")
    # f_dict = {"MAP" : [70, 75], "PP" : [45, 55]}
    # plot_inverse_feature_with_confidence(data, f_dict, ax=axes[3])
    # print("4")
    # f_dict = {"MAP" : [70, 75], "PP" : [55, 65]}
    # plot_inverse_feature_with_confidence(data, f_dict, ax=axes[4])
    # print("5")


    # f_dict = {"MAP" : [75, 80]}
    # plot_inverse_feature_with_confidence(data, f_dict, ax=axes[0])
    # print("6")
    # f_dict = {"MAP" : [75, 80], "PP" : [25, 35]}
    # plot_inverse_feature_with_confidence(data, f_dict, ax=axes[1])
    # print("7")
    # f_dict = {"MAP" : [75, 80], "PP" : [35, 45]}
    # plot_inverse_feature_with_confidence(data, f_dict, ax=axes[2])
    # print("8")
    # f_dict = {"MAP" : [75, 80], "PP" : [45, 55]}
    # plot_inverse_feature_with_confidence(data, f_dict, ax=axes[3])
    # print("9")
    # f_dict = {"MAP" : [75, 80], "PP" : [55, 65]}
    # plot_inverse_feature_with_confidence(data, f_dict, ax=axes[4])
    # print("10")


    # f_dict = {"MAP" : [80, 85]}
    # plot_inverse_feature_with_confidence(data, f_dict, ax=axes[0])
    # print("11")
    # f_dict = {"MAP" : [80, 85], "PP" : [25, 35]}
    # plot_inverse_feature_with_confidence(data, f_dict, ax=axes[1])
    # print("12")
    # f_dict = {"MAP" : [80, 85], "PP" : [35, 45]}
    # plot_inverse_feature_with_confidence(data, f_dict, ax=axes[2])
    # print("13")
    # f_dict = {"MAP" : [80, 85], "PP" : [45, 55]}
    # plot_inverse_feature_with_confidence(data, f_dict, ax=axes[3])
    # print("14")
    # f_dict = {"MAP" : [80, 85], "PP" : [55, 65]}
    # plot_inverse_feature_with_confidence(data, f_dict, ax=axes[4])
    # print("15")


    # f_dict = {"MAP" : [85, 90]}
    # plot_inverse_feature_with_confidence(data, f_dict, ax=axes[0])
    # print("16")
    # f_dict = {"MAP" : [85, 90], "PP" : [25, 35]}
    # plot_inverse_feature_with_confidence(data, f_dict, ax=axes[1])
    # print("17")
    # f_dict = {"MAP" : [85, 90], "PP" : [35, 45]}
    # plot_inverse_feature_with_confidence(data, f_dict, ax=axes[2])
    # print("18")
    # f_dict = {"MAP" : [85, 90], "PP" : [45, 55]}
    # plot_inverse_feature_with_confidence(data, f_dict, ax=axes[3])
    # print("19")
    # f_dict = {"MAP" : [85, 90], "PP" : [55, 65]}
    # plot_inverse_feature_with_confidence(data, f_dict, ax=axes[4])
    # print("20")

    # f_dict = {"MAP" : [90, 95]}
    # plot_inverse_feature_with_confidence(data, f_dict, ax=axes[0])
    # print("16")
    # f_dict = {"MAP" : [90, 95], "PP" : [25, 35]}
    # plot_inverse_feature_with_confidence(data, f_dict, ax=axes[1])
    # print("17")
    # f_dict = {"MAP" : [90, 95], "PP" : [35, 45]}
    # plot_inverse_feature_with_confidence(data, f_dict, ax=axes[2])
    # print("18")
    # f_dict = {"MAP" : [90, 95], "PP" : [45, 55]}
    # plot_inverse_feature_with_confidence(data, f_dict, ax=axes[3])
    # print("19")
    # f_dict = {"MAP" : [90, 95], "PP" : [55, 65]}
    # plot_inverse_feature_with_confidence(data, f_dict, ax=axes[4])
    # print("20")

    f_dict = {"MAP" : [95, 100]}
    plot_inverse_feature_with_confidence(data, f_dict, ax=axes[0])
    print("16")
    f_dict = {"MAP" : [95, 100], "PP" : [25, 35]}
    plot_inverse_feature_with_confidence(data, f_dict, ax=axes[1])
    print("17")
    f_dict = {"MAP" : [95, 100], "PP" : [35, 45]}
    plot_inverse_feature_with_confidence(data, f_dict, ax=axes[2])
    print("18")
    f_dict = {"MAP" : [95, 100], "PP" : [45, 55]}
    plot_inverse_feature_with_confidence(data, f_dict, ax=axes[3])
    print("19")
    f_dict = {"MAP" : [95, 100], "PP" : [55, 65]}
    plot_inverse_feature_with_confidence(data, f_dict, ax=axes[4])
    print("20")



    plt.tight_layout()
    plt.show()


def compute_min_max(data, pp_intervals, map_confidence=2.5):

    # poiščemo vse točke okoli map in znotrj vsakega pp intervala
    results = dict()

    # map_values should be all values from 60 to 120

    map_values = [i for i in range(60, 120)]

    for map_val in map_values:

        print(map_val)

        map_lower = map_val - map_confidence
        map_upper = map_val + map_confidence

        middle_results = dict((i, []) for i in range(len(pp_intervals)))

        for i in range(len(pp_intervals)): # in pp_intervals:

            pp_interval = pp_intervals[i]

            pp_lower = pp_interval[0]
            pp_upper = pp_interval[1]

            

            for pt in data:

                part = pt['part']
                map_value = 1/3 * (np.max(part) - np.min(part)) + np.min(part)
                pp = np.max(part) - np.min(part)

                if map_value >= map_lower and map_value <= map_upper and pp >= pp_lower and pp <= pp_upper:
                    middle_results[i].append(pt['params'])


        results[map_val] = middle_results

    # save results to a file
    with open("data/min_max_results.json", "w") as json_file:
        json.dump(results, json_file)

    return results

def plot_min_max(filepath="data/min_max_results.json"):

    with open(filepath, "r") as json_file:
        results = json.load(json_file)

    num_plots = len(results[list(results.keys())[0]])

    rs, cs, ss = [], [], []

    for i in range(num_plots):

        pp_lists = dict()

        for map_val in results:

            pp_lists[map_val] = results[map_val][str(i)]

        min_rs = [min([elt[1] for elt in pp_lists[map_val]]) if len([elt[1] for elt in pp_lists[map_val]]) != 0 else 0 for map_val in pp_lists]
        max_rs = [max([elt[1] for elt in pp_lists[map_val]]) if len([elt[1] for elt in pp_lists[map_val]]) != 0 else 0 for map_val in pp_lists]
        mean_rs = [np.mean([elt[1] for elt in pp_lists[map_val]]) if len([elt[1] for elt in pp_lists[map_val]]) != 0 else 0 for map_val in pp_lists]
        std_rs = [np.std([elt[1] for elt in pp_lists[map_val]]) if len([elt[1] for elt in pp_lists[map_val]]) != 0 else 0 for map_val in pp_lists]
        std_rs_upper = [mean_rs[i] + std_rs[i] for i in range(len(mean_rs))]
        std_rs_lower = [mean_rs[i] - std_rs[i] for i in range(len(mean_rs))]
        
        # max_rs = [max([elt[1] for elt in pp_lists[map_val]]) for map_val in pp_lists]

        min_cs = [min([elt[0] for elt in pp_lists[map_val]]) if len([elt[0] for elt in pp_lists[map_val]]) != 0 else 0 for map_val in pp_lists]
        max_cs = [max([elt[0] for elt in pp_lists[map_val]]) if len([elt[0] for elt in pp_lists[map_val]]) != 0 else 0 for map_val in pp_lists]
        mean_cs = [np.mean([elt[0] for elt in pp_lists[map_val]]) if len([elt[0] for elt in pp_lists[map_val]]) != 0 else 0 for map_val in pp_lists]
        std_cs = [np.std([elt[0] for elt in pp_lists[map_val]]) if len([elt[0] for elt in pp_lists[map_val]]) != 0 else 0 for map_val in pp_lists]

        std_cs_upper = [mean_cs[i] + std_cs[i] for i in range(len(mean_cs))]
        std_cs_lower = [mean_cs[i] - std_cs[i] for i in range(len(mean_cs))]

        min_ss = [min([elt[2] for elt in pp_lists[map_val]]) if len([elt[2] for elt in pp_lists[map_val]]) != 0 else 0 for map_val in pp_lists]
        max_ss = [max([elt[2] for elt in pp_lists[map_val]]) if len([elt[2] for elt in pp_lists[map_val]]) != 0 else 0 for map_val in pp_lists]
        mean_ss = [np.mean([elt[2] for elt in pp_lists[map_val]]) if len([elt[2] for elt in pp_lists[map_val]]) != 0 else 0 for map_val in pp_lists]
        std_ss = [np.std([elt[2] for elt in pp_lists[map_val]]) if len([elt[2] for elt in pp_lists[map_val]]) != 0 else 0 for map_val in pp_lists]

        std_ss_upper = [mean_ss[i] + std_ss[i] for i in range(len(mean_ss))]
        std_ss_lower = [mean_ss[i] - std_ss[i] for i in range(len(mean_ss))]

        rs.append((min_rs, max_rs, mean_rs, std_rs_upper, std_rs_lower))
        cs.append((min_cs, max_cs, mean_cs, std_cs_upper, std_cs_lower))
        ss.append((min_ss, max_ss, mean_ss, std_ss_upper, std_ss_lower))

    fig, axes = plt.subplots(1, num_plots, figsize=(12,6))

    x_axis = []
    for map_val in results:
        if int(map_val) <= 100:
            x_axis.append(int(map_val))

    for i in range(num_plots):

        axes[i].plot(x_axis, rs[i][0][:len(x_axis)], label="min", color='blue')
        axes[i].plot(x_axis, rs[i][1][:len(x_axis)], label="max", color='blue')
        axes[i].plot(x_axis, rs[i][2][:len(x_axis)], label="mean", color='red')
        axes[i].plot(x_axis, rs[i][3][:len(x_axis)], label="std", color='red', linestyle='dashed', alpha=0.5)
        axes[i].plot(x_axis, rs[i][4][:len(x_axis)], label="std", color='red', linestyle='dashed', alpha=0.5)
        axes[i].fill_between(x_axis, rs[i][3][:len(x_axis)], rs[i][4][:len(x_axis)], color='red', alpha=0.2)
        axes[i].set_title(f"UPORNOST \n PP interval: {pp_intervals[i]}")
        axes[i].set_ylim(0.6, 1.6)

        # axes[i].plot(x_axis, cs[i][0][:len(x_axis)], label="min", color='blue')
        # axes[i].plot(x_axis, cs[i][1][:len(x_axis)], label="max", color='blue')
        # axes[i].plot(x_axis, cs[i][2][:len(x_axis)], label="mean", color='red')
        # axes[i].plot(x_axis, cs[i][3][:len(x_axis)], label="std", color='red', linestyle='dashed', alpha=0.5)
        # axes[i].plot(x_axis, cs[i][4][:len(x_axis)], label="std", color='red', linestyle='dashed', alpha=0.5)
        # axes[i].fill_between(x_axis, cs[i][3][:len(x_axis)], cs[i][4][:len(x_axis)], color='red', alpha=0.2)
        # axes[i].set_title(f"KONTRAKTILNOST \n PP interval: {pp_intervals[i]}")
        # axes[i].set_ylim(0.5, 2.2)

        # axes[i].plot(x_axis, ss[i][0][:len(x_axis)], label="min", color='blue')
        # axes[i].plot(x_axis, ss[i][1][:len(x_axis)], label="max", color='blue')
        # axes[i].plot(x_axis, ss[i][2][:len(x_axis)], label="mean", color='red')
        # axes[i].plot(x_axis, ss[i][3][:len(x_axis)], label="std", color='red', linestyle='dashed', alpha=0.5)
        # axes[i].plot(x_axis, ss[i][4][:len(x_axis)], label="std", color='red', linestyle='dashed', alpha=0.5)
        # axes[i].fill_between(x_axis, ss[i][3][:len(x_axis)], ss[i][4][:len(x_axis)], color='red', alpha=0.2)
        # axes[i].set_title(f"PODAJNOST \n PP interval: {pp_intervals[i]}")
        # axes[i].set_ylim(0, 1.5)

    # show every tenth x tick
    # for ax in axes:
    #     for label in ax.get_xticklabels():
    #         label.set_rotation(45)
    #         label.set_horizontalalignment('right')
    #     ax.set_xticks(ax.get_xticks()[::10])

    plt.tight_layout()
    plt.show()


def best_features(df, feature_name, std, min_val, max_val, min_map=None, max_map=None, min_pp=None, max_pp=None, filepath=None):

    # create 100 pts between min and max

    x = np.linspace(min_val, max_val, 100)

    min_cs_1, max_cs_1, mean_cs_1 = [], [], []
    min_rs_1, max_rs_1, mean_rs_1 = [], [], []
    min_ss_1, max_ss_1, mean_ss_1 = [], [], []

    min_cs_2, max_cs_2, mean_cs_2 = [], [], []
    min_rs_2, max_rs_2, mean_rs_2 = [], [], []
    min_ss_2, max_ss_2, mean_ss_2 = [], [], []

    min_cs_4, max_cs_4, mean_cs_4 = [], [], []
    min_rs_4, max_rs_4, mean_rs_4 = [], [], []
    min_ss_4, max_ss_4, mean_ss_4 = [], [], []

    if min_map is not None and max_map is not None:

        df = df[(df['MAP'] >= min_map) & (df['MAP'] <= max_map)]

    if min_pp is not None and max_pp is not None:

        df = df[(df['PP'] >= min_pp) & (df['PP'] <= max_pp)]


    for val in x:
        x_lower1, x_upper1 = val - std, val + std
        x_lower2, x_upper2 = val - std/2, val + std/2
        # x_lower3, x_upper3 = val - std/3, val + std/3
        x_lower4, x_upper4 = val - std/4, val + std/4

        pts1 = df[(df[feature_name] >= x_lower1) & (df[feature_name] <= x_upper1)]
        pts2 = df[(df[feature_name] >= x_lower2) & (df[feature_name] <= x_upper2)]
        # pts3 = df[(df[feature_name] >= x_lower3) & (df[feature_name] <= x_upper3)]
        pts4 = df[(df[feature_name] >= x_lower4) & (df[feature_name] <= x_upper4)]

        min_c1, max_c1 = np.min(pts1['C']), np.max(pts1['C'])
        mean_c1 = np.mean(pts1['C'])
        min_r1, max_r1 = np.min(pts1['R']), np.max(pts1['R'])
        mean_r1 = np.mean(pts1['R'])
        min_s1, max_s1 = np.min(pts1['S']), np.max(pts1['S'])
        mean_s1 = np.mean(pts1['S'])

        min_cs_1.append(min_c1)
        max_cs_1.append(max_c1)
        mean_cs_1.append(mean_c1)
        min_rs_1.append(min_r1)
        max_rs_1.append(max_r1)
        mean_rs_1.append(mean_r1)
        min_ss_1.append(min_s1)
        max_ss_1.append(max_s1)
        mean_ss_1.append(mean_s1)

        min_c2, max_c2 = np.min(pts2['C']), np.max(pts2['C'])
        mean_c2 = np.mean(pts2['C'])
        min_r2, max_r2 = np.min(pts2['R']), np.max(pts2['R'])
        mean_r2 = np.mean(pts2['R'])
        min_s2, max_s2 = np.min(pts2['S']), np.max(pts2['S'])
        mean_s2 = np.mean(pts2['S'])

        min_cs_2.append(min_c2)
        max_cs_2.append(max_c2)
        mean_cs_2.append(mean_c2)
        min_rs_2.append(min_r2)
        max_rs_2.append(max_r2)
        mean_rs_2.append(mean_r2)
        min_ss_2.append(min_s2)
        max_ss_2.append(max_s2)
        mean_ss_2.append(mean_s2)

        # min_c3, max_c3 = np.min(pts3['C']), np.max(pts3['C'])
        # min_r3, max_r3 = np.min(pts3['R']), np.max(pts3['R'])
        # min_s3, max_s3 = np.min(pts3['S']), np.max(pts3['S'])

        min_c4, max_c4 = np.min(pts4['C']), np.max(pts4['C'])
        mean_c4 = np.mean(pts4['C'])
        min_r4, max_r4 = np.min(pts4['R']), np.max(pts4['R'])
        mean_r4 = np.mean(pts4['R'])
        min_s4, max_s4 = np.min(pts4['S']), np.max(pts4['S'])
        mean_s4 = np.mean(pts4['S'])

        min_cs_4.append(min_c4)
        max_cs_4.append(max_c4)
        mean_cs_4.append(mean_c4)
        min_rs_4.append(min_r4)
        max_rs_4.append(max_r4)
        mean_rs_4.append(mean_r4)
        min_ss_4.append(min_s4)
        max_ss_4.append(max_s4)
        mean_ss_4.append(mean_s4)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    axes[0].plot(x, min_cs_1, label="min", color='blue', alpha=0.5)
    axes[0].plot(x, max_cs_1, label="max", color='blue', alpha=0.5)
    axes[0].plot(x, mean_cs_1, label="mean", color='blue', linestyle='dashed')

    axes[0].plot(x, min_cs_2, label="min", color='red', alpha=0.5)
    axes[0].plot(x, max_cs_2, label="max", color='red', alpha=0.5)
    axes[0].plot(x, mean_cs_2, label="mean", color='red', linestyle='dashed')

    axes[0].plot(x, min_cs_4, label="min", color='green', alpha=0.5)
    axes[0].plot(x, max_cs_4, label="max", color='green', alpha=0.5)
    axes[0].plot(x, mean_cs_4, label="mean", color='green', linestyle='dashed')

    axes[0].set_title("Kontraktilnost")
    # set yaxis range
    axes[0].set_ylim(0.5, 2.2)

    axes[1].plot(x, min_rs_1, label="min", color='blue', alpha=0.5)
    axes[1].plot(x, max_rs_1, label="max", color='blue', alpha=0.5)
    axes[1].plot(x, mean_rs_1, label="mean", color='blue', linestyle='dashed')

    axes[1].plot(x, min_rs_2, label="min", color='red', alpha=0.5)
    axes[1].plot(x, max_rs_2, label="max", color='red', alpha=0.5)
    axes[1].plot(x, mean_rs_2, label="mean", color='red', linestyle='dashed')

    axes[1].plot(x, min_rs_4, label="min", color='green', alpha=0.5)
    axes[1].plot(x, max_rs_4, label="max", color='green', alpha=0.5)
    axes[1].plot(x, mean_rs_4, label="mean", color='green', linestyle='dashed')

    axes[1].set_title("Upornost")
    axes[1].set_ylim(0.6, 1.6)

    axes[2].plot(x, min_ss_1, label="min", color='blue', alpha=0.5)
    axes[2].plot(x, max_ss_1, label="max", color='blue', alpha=0.5)
    axes[2].plot(x, mean_ss_1, label="mean", color='blue', linestyle='dashed')

    axes[2].plot(x, min_ss_2, label="min", color='red', alpha=0.5)
    axes[2].plot(x, max_ss_2, label="max", color='red', alpha=0.5)
    axes[2].plot(x, mean_ss_2, label="mean", color='red', linestyle='dashed')

    axes[2].plot(x, min_ss_4, label="min", color='green', alpha=0.5)
    axes[2].plot(x, max_ss_4, label="max", color='green', alpha=0.5)
    axes[2].plot(x, mean_ss_4, label="mean", color='green', linestyle='dashed')

    axes[2].set_title("Podajnost")
    axes[2].set_ylim(0, 1.5)

    plt.tight_layout()
    plt.savefig(filepath)

    
    # plt.show()









if __name__=="__main__":

    data, stable_part = read_parsed_data()
    # plot_bp_vs_r(data)

    X, C, R, S, feature_names, df = load_new_data()
    # nearby_pts(df, "čas do SP", plot=True)

    # plot_bp_vs_r(data, param="R", mode="dbp_vs_sbp", blue_scale=blue_scale_r)
    # plot_bp_vs_r(data, param="C", mode="dbp_vs_sbp", blue_scale=blue_scale_c)
    # plot_bp_vs_r(data, param="S", mode="dbp_vs_sbp", blue_scale=blue_scale_s)

    # plot_bp_vs_r(data, param="R", mode="map_vs_pp", blue_scale=blue_scale_r)
    # plot_bp_vs_r(data, param="C", mode="map_vs_pp", blue_scale=blue_scale_c)
    # plot_bp_vs_r(data, param="S", mode="map_vs_pp", blue_scale=blue_scale_s)

    # feature_dict = {
    #     # "DBP" : [0, 80],
    #     # "SBP" : [0, 120],
    #     "MAP" : [70, 75]
    # }

    map_list = [[60, 65], [65, 70], [70, 75], [75, 80], [80, 85], [85, 90], [90, 95], [95, 100]]

    # for elt in map_list:
    #     feature_dict = {
    #         "MAP" : elt
    #     }

    #     plot_inverse_feature_with_confidence(df, feature_dict)

    feature_dict = {
        "MAP" : [[60, 65], [65, 70], [70, 75], [75, 80], [80, 85], [85, 90], [90, 95], [95, 100]],
        # "PP" : [[35, 45], [35, 45], [35, 45], [35, 45], [50, 60], [50, 60], [50, 60], [50, 60]],
    }

    # plot_multiple_plots(df, feature_dict)
    # plot_map_progression_with_pp(df)

    # plot_waves(data, 80, 100)

    # f_dict = {"MAP" : [70, 75]}
    # plot_inverse_feature_with_confidence(data, f_dict,)

    pp_intervals = [[25, 35], [35, 45], [45, 55], [55, 65]]

    # compute_min_max(data, pp_intervals, map_confidence=2.5)
    # plot_min_max()

    feature_dict = { # velikostni red 
        't_sp' : [0.02, 0.1452, 0.3443], # 0.02
        'max_slope' : [0.31, 0.4862, 2.7185], # 1
        'bpdn_pp_ratio' : [0.38, 1.5467, 3.0784], # 0.5
        't_start_dn' : [0.01, 0.2131, 0.3911], # 500
        'dbp' : [19.78, 22.4922, 120],
        # 'map' : 21.6,
        # 'pp' : 8.81,
    }

    renaming2 = dict((renaming[key], key) for key in renaming)
    feature_dict2 = dict((renaming2[feature], feature_dict[feature]) for feature in feature_dict)

    for feature_name in feature_dict:
        name = renaming2[feature_name]
        std = feature_dict[feature_name][0]
        min_val = feature_dict[feature_name][1]
        max_val = feature_dict[feature_name][2]
        best_features(df, name, std, min_val=min_val, max_val=max_val, filepath=f"graphical_approach/{feature_name}.png")

    map_values = [[70, 80], [80, 90], [90, 100]]
    pp_values = [[25, 35], [35, 45], [45, 55], [55, 65]]

    for map_val in map_values:
        for feature_name in feature_dict:
            name = renaming2[feature_name]
            std = feature_dict[feature_name][0]
            min_val = feature_dict[feature_name][1]
            max_val = feature_dict[feature_name][2]
            min_map = map_val[0]
            max_map = map_val[1]
            best_features(df, name, std, min_val=min_val, max_val=max_val, min_map=min_map, max_map=max_map, filepath=f"graphical_approach/{feature_name}_map_{min_map}_{max_map}.png")

    for map_val in map_values:
        for pp_val in pp_values:
            for feature_name in feature_dict:
                name = renaming2[feature_name]
                std = feature_dict[feature_name][0]
                min_val = feature_dict[feature_name][1]
                max_val = feature_dict[feature_name][2]
                min_map = map_val[0]
                max_map = map_val[1]
                min_pp = pp_val[0]
                max_pp = pp_val[1]
                best_features(df, name, std, min_val=min_val, max_val=max_val, min_map=min_map, max_map=max_map, min_pp=min_pp, max_pp=max_pp, filepath=f"graphical_approach/{feature_name}_map_{min_map}_{max_map}_pp_{min_pp}_{max_pp}.png")

    