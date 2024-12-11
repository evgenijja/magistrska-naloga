import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from feature_renaming import renaming

from itertools import product

import json

"""
Based on the choice of features plot inverse plot.
"""

def read_parsed_data(filter=False):

    with open("data/parsed_data2.json", "r") as json_file:
        data = json.load(json_file)

    print("Število simulacij:", len(data))

    new_data, stable_part = [], []
    for point in data:
        part = point['part']
        if point['params'] == [1, 1, 1]:
            stable_part.append(point)
        if not np.isnan(part).any() and part != len(part) * [0] and len(set(part)) > 1 and sum([0 if elt >= 0 else 1 for elt in part]) == 0:
            if point not in new_data:
                new_data.append(point)

    return new_data



def load_new_data (filepath="data/features_FINAL.xlsx"):

    # read dataframe that has and index column and header
    df = pd.read_excel(filepath, header=0, index_col=0)
    print(len(df))
    df.dropna(inplace=True)
    print(len(df))
    C, R, S = df['C'], df['R'], df['S']
    
    # X is all the columns that are not C, R, S
    feature_names = df.columns.tolist()
    # feature_names.remove('C')
    # feature_names.remove('R')
    # feature_names.remove('S')
    # X = df[feature_names]

    return df, feature_names

def load_data_for_residuum_features(filepath1="data/residuum_features.xlsx", filepath2="data/features_FINAL.xlsx"):

    # read dataframe that has and index column and header
    df1 = pd.read_excel(filepath1, header=0, index_col=0)
    df2 = pd.read_excel(filepath2, header=0, index_col=0)

    merged_df = pd.merge(df1, df2, on=["C", "R", "S"], how="inner")

    merged_df = merged_df.rename(columns=renaming)

    return merged_df


def get_inverse_plot_data(data, feature_dict, reference_val, shape=True):

    all_points_to_plot = []

    for feature in feature_dict:

        confidence = feature_dict[feature]

        # reference_feature = list(data[(round(data["C"], 2) == reference_pt["C"]) & (round(data["R"], 2) == reference_pt["R"]) & (round(data["S"], 2) == reference_pt["S"])][feature])[0]
        # print(f"reference: {reference_feature}")
        # print(f"reference: {reference_feature}")
        lower_bound = reference_val - confidence
        upper_bound = reference_val + confidence

        # get all points in the confidence interval
        points = data[(data[feature] >= lower_bound) & (data[feature] <= upper_bound)]

        C, R, S = [elt for elt in points["C"]], [elt for elt in points["R"]], [elt for elt in points["S"]]
        points_to_plot = [(C[i], R[i], S[i]) for i in range(len(C))]
        # print(points_to_plot)

        all_points_to_plot.append(points_to_plot)


    # all_points_to_plot is a list of lists of points with three coordinates. I want to find the intersection of all lists
    # i only want points that are in all lists
    # print(all_points_to_plot[0])
    # intersection = set(all_points_to_plot[0])
    intersection = set(all_points_to_plot[0])
    for i in range(1, len(all_points_to_plot)):
        intersection = intersection.intersection(set(all_points_to_plot[i]))

    intersection = list(intersection)

    print(len(intersection))

    x_coords = [point[0] for point in intersection]
    y_coords = [point[1] for point in intersection]
    z_coords = [point[2] for point in intersection]

    # shape
    if len(x_coords) == 0:
        return None
    if max(x_coords) - min(x_coords) <= 0.4: # 1/4 of the range
        x_shape = "narrow"
    else:
        x_shape = "wide"

    if max(y_coords) - min(y_coords) <= 0.2: # 1/4 of the range
        y_shape = "narrow"
    else:
        y_shape = "wide"

    if max(z_coords) - min(z_coords) <= 0.3: # 1/4 of the range
        z_shape = "narrow"
    else:
        z_shape = "wide"
    
    shapes = [x_shape, y_shape, z_shape]
    wides = shapes.count("wide")
    narrows = shapes.count("narrow")

    if wides == 3:
        shape = "prostor"
    elif wides == 2 and narrows == 1:
        shape = "ploskev"
    elif wides == 1 and narrows == 2:
        shape = "valj"
    elif narrows == 3:
        shape = "točka"

    if shape: 
        return shape 
    
    else: 
        return len(x_coords)#, y_coords, z_coords


def plot_inverse_single_reference_pt(ax, data, feature_dict, reference_pt, refernce_pt_name=""):
    """
    df .. dataframe with features
    reference_pts .. list of dicts with pts: {'C': 1.1, 'R': 2.1, 'S': 0.9}
    feature dict .. keys are ORIGINAL feature names, values are confidence values
    horizontal .. plot 2 rows 4 plots, else plot 4 rows 2 plots
    """

    all_points_to_plot = []

    for feature in feature_dict:

        confidence = feature_dict[feature]

        reference_feature = list(data[(round(data["C"], 2) == reference_pt["C"]) & (round(data["R"], 2) == reference_pt["R"]) & (round(data["S"], 2) == reference_pt["S"])][feature])[0]
        print(f"reference: {reference_feature}")
        # print(f"reference: {reference_feature}")
        lower_bound = reference_feature - confidence
        upper_bound = reference_feature + confidence

        # get all points in the confidence interval
        points = data[(data[feature] >= lower_bound) & (data[feature] <= upper_bound)]

        C, R, S = [elt for elt in points["C"]], [elt for elt in points["R"]], [elt for elt in points["S"]]
        points_to_plot = [(C[i], R[i], S[i]) for i in range(len(C))]
        # print(points_to_plot)

        all_points_to_plot.append(points_to_plot)


    # all_points_to_plot is a list of lists of points with three coordinates. I want to find the intersection of all lists
    # i only want points that are in all lists
    # print(all_points_to_plot[0])
    # intersection = set(all_points_to_plot[0])
    intersection = set(all_points_to_plot[0])
    for i in range(1, len(all_points_to_plot)):
        intersection = intersection.intersection(set(all_points_to_plot[i]))

    intersection = list(intersection)

    print(len(intersection))

    x_coords = [point[0] for point in intersection]
    y_coords = [point[1] for point in intersection]
    z_coords = [point[2] for point in intersection]

    # shape
    if np.max(x_coords) - np.min(x_coords) <= 0.4: # 1/4 of the range
        x_shape = "narrow"
    else:
        x_shape = "wide"

    if np.max(y_coords) - np.min(y_coords) <= 0.2: # 1/4 of the range
        y_shape = "narrow"
    else:
        y_shape = "wide"

    if np.max(z_coords) - np.min(z_coords) <= 0.3: # 1/4 of the range
        z_shape = "narrow"
    else:
        z_shape = "wide"
    
    shapes = [x_shape, y_shape, z_shape]
    wides = shapes.count("wide")
    narrows = shapes.count("narrow")

    if wides == 3:
        shape = "prostor"
    elif wides == 2 and narrows == 1:
        shape = "ploskev"
    elif wides == 1 and narrows == 2:
        shape = "valj"
    elif narrows == 3:
        shape = "točka"


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # plot one extra point in black and thicker
    

    ax.scatter(x_coords, y_coords, z_coords)
    ax.scatter(reference_pt["C"], reference_pt["R"], reference_pt["S"], color='k', s=15)

    ax.set_xlabel('kontraktilnost')
    ax.set_ylabel('upornost')
    ax.set_zlabel('podajnost')

    # ax.set_title(f"Število točk: {len(x_coords)}, Oblika: {shape} \n C: [{round(np.min(x_coords), 2)}, {round(np.max(x_coords), 2)}],\n Upornost: [{round(np.min(y_coords), 2)}, {round(np.max(y_coords), 2)}], \n Podajnost: [{round(np.min(z_coords), 2)}, {round(np.max(z_coords), 2)}]")

    ax.set_title(f"Referenčna točka {refernce_pt_name}")

    # ranges: 
    # C : 0.55 - 2.15
    # R : 0.7 - 1.5
    # S :  0.1 - 1.3

    ax.set_xlim((0.55, 2.15))
    ax.set_ylim((0.7, 1.5))
    ax.set_zlim((0.1, 1.3))

    plt.tight_layout()

    # return points that we plotted
    return intersection

    # plt.show()

def plot_inverse_3d_and_2d(data, parts, feature_dict, reference_pts):
    """
    parts is a list of dicts with params and parts
    """

    # make subplots (len(reference_pts) rows and 3 columns)

    

    

    fig, axs = plt.subplots(2, len(reference_pts), figsize=(20, 10))

    # for each reference point we plot 3 subplots in the same row: the first one is the 3d plot plot_single_plot, the second and the third are 2d plots

    for g in range(len(reference_pts)):

        all_points_to_plot = []
        intersection_parts = []

        reference_pt = reference_pts[g]

        for feature in feature_dict:

            confidence = feature_dict[feature]

            reference_feature = list(data[(round(data["C"], 2) == reference_pt["C"]) & (round(data["R"], 2) == reference_pt["R"]) & (round(data["S"], 2) == reference_pt["S"])][feature])[0]
            print(f"reference: {reference_feature}")
            # print(f"reference: {reference_feature}")
            lower_bound = reference_feature - confidence
            upper_bound = reference_feature + confidence

            # get all points in the confidence interval
            points = data[(data[feature] >= lower_bound) & (data[feature] <= upper_bound)]

            C, R, S = [elt for elt in points["C"]], [elt for elt in points["R"]], [elt for elt in points["S"]]
            points_to_plot = [(C[i], R[i], S[i]) for i in range(len(C))]
            # print(points_to_plot)

            all_points_to_plot.append(points_to_plot)

        intersection = set(all_points_to_plot[0])
        for i in range(1, len(all_points_to_plot)):
            intersection = intersection.intersection(set(all_points_to_plot[i]))

        # points to plot
        intersection = list(intersection)
        print(len(intersection))

        part_to_plot = []
        for elt in parts:

            params = elt["params"]
            part = elt["part"]

            for elt_int in intersection:
                if round(params[0], 2) == elt_int[0] and round(params[1], 2) == elt_int[1] and round(params[2], 2) == elt_int[2]:
                    part_to_plot.append(part)

        # if g % 3 == 0:
        # ax = axs[0, g]

        # ax = fig.add_subplot(111, projection='3d')

        # # plot 3d plot
        # x_coords = [point[0] for point in intersection]
        # y_coords = [point[1] for point in intersection]
        # z_coords = [point[2] for point in intersection]

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

        # ax.scatter(x_coords, y_coords, z_coords)
        # # ax.scatter(reference_pt["C"], reference_pt["R"], reference_pt["S"], color='k', s=15)

        # ax.set_xlabel('kontraktilnost')
        # ax.set_ylabel('upornost')
        # ax.set_zlabel('podajnost')

        # ax.set_title(f"Število točk: {len(x_coords)}, Oblika: {shape} \n C: [{round(np.min(x_coords), 2)}, {round(np.max(x_coords), 2)}],\n Upornost: [{round(np.min(y_coords), 2)}, {round(np.max(y_coords), 2)}], \n Podajnost: [{round(np.min(z_coords), 2)}, {round(np.max(z_coords), 2)}]")

        # ax.set_xlim((0.55, 2.15))
        # ax.set_ylim((0.7, 1.5))
        # ax.set_zlim((0.1, 1.3))

        # elif g % 3 == 1:
        x, y = [], []
        ax = axs[0, g]
        for part in part_to_plot:
            
            
            MAP = 1/3 * (np.max(part) - np.min(part)) + np.min(part)
            PP = np.max(part) - np.min(part)

            x.append(PP)
            y.append(MAP)

        ax.scatter(x, y)
        ax.set_xlabel('PP')
        ax.set_ylabel('MAP')
        ax.set_title('PP in MAP')

        # set axis limit
        ax.set_xlim((0, 150))
        ax.set_ylim((0, 150))

        # elif g % 3 == 2:
        ax = axs[1, g]
        x, y = [], []
        for part in part_to_plot:
            

            dbp = np.min(part)
            sbp = np.max(part)

            x.append(dbp)
            y.append(sbp)

        ax.scatter(x, y)
        ax.set_xlabel('dbp')
        ax.set_ylabel('sbp')
        ax.set_title('dbp in sbp')

        # set axis limit
        ax.set_xlim((0, 150))
        ax.set_ylim((0, 150))


    plt.tight_layout()
    plt.show()

def inverse_shape(data, feature_dict):

    for elt in data:
        pass

    

def multple_reference_plots(data, reference_pts, feature_dict, filepath):

    # reference_pts is a list of 8 dicts
    # i want to create subplots with 2 rows and 4 columns
    # each subplot will be a plot_inverse_single_reference_pt for one reference point

    # fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), subplot_kw={'projection': '3d'})

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # for ax, Z, title in zip(axes, [Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8],  ['Sin', 'Cos', 'Tan', 'Exp', 'Sqrt', 'Log', 'Sinh', 'Cosh']):
    #     plot_3d(ax, X, Y, Z, title)

    # for reference_pt, ax in zip(reference_pts, axes):
    for i in range(len(reference_pts)):
        reference_pt = reference_pts[i]
        ax = axes[i]
        plot_inverse_single_reference_pt(ax, data, feature_dict, reference_pt, refernce_pt_name=f"{i+1}")

    plt.tight_layout()
    plt.savefig(filepath)
    plt.show()

def color_pts_by_nearby_pts(df, feature_dict):
    """
    Za vsako točko hočemo pogledati kako velik je inverzni prostor okoli nje in kakšne oblike je

    """
    for feature in feature_dict:
        # for each row we compute how many other rows have the value that is within the confidence interval
        confidence = feature_dict[feature]
        
        df[f"{feature}_count"] = df[feature].apply(lambda x: ((df[feature] - x).abs() <= confidence).sum())
        df[f"{feature}_shape"] = df[feature].apply(lambda x: get_inverse_plot_data(df, feature_dict, x, shape=True))
        df[f"{feature}_pts_count"] = df[feature].apply(lambda x: get_inverse_plot_data(df, feature_dict, x, shape=False))

    return df

def compute_all_maps_and_pps(data):

    results_map, results_pp = dict(), dict()
    for point in data:
        C, R, S = point["params"]
        part = point["part"]
        MAP = round(1/3 * (np.max(part) - np.min(part)) + np.min(part))
        PP = round(np.max(part) - np.min(part))

        if MAP not in results_map:
            results_map[MAP] = [[round(C, 2), round(R, 2), round(S, 2)]]
        else:
            results_map[MAP].append([round(C, 2), round(R, 2), round(S, 2)])

        if PP not in results_pp:
            results_pp[PP] = [[round(C, 2), round(R, 2), round(S, 2)]]
        else:
            results_pp[PP].append([round(C, 2), round(R, 2), round(S, 2)])

    # order the results by keys
    sorted_results_map = dict(sorted(results_map.items()))
    sorted_results_pp = dict(sorted(results_pp.items()))

    return sorted_results_map, sorted_results_pp
                                  



def inverse_problem_count(maps, pps, pp_interval=[20, 25]):
    """
    Zanima nas ali lahko iz MAP in PP napovemo upornost in kdaj lahko napovemo kontraktilnost
    """
    results_r, results_c, results_s = dict(), dict(), dict()
    for map_value in maps:

        lower_bound, upper_bound = map_value - 5, map_value + 5

        map_pts, pp_pts = [], []

        for i in range(lower_bound, upper_bound):
            if i in maps:
                for elt in maps[i]:
                    map_pts.append(elt)

        for i in range(pp_interval[0], pp_interval[1]):
            if i in pps:
                for elt in pps[i]:
                    pp_pts.append(elt)

        pts = []
        for pt in map_pts:
            if pt in pp_pts:
                pts.append(pt)

        print(len(map_pts), len(pp_pts), len(pts))


            # for j in range(pp_interval[0], pp_interval[1]):
            #     if i in maps and j in pps:
            #         pts.append(maps[i] + pps[j])

        if len(pts) > 0:
            r_min, r_max = min([elt[1] for elt in pts]), max([elt[1] for elt in pts])
            c_min, c_max = min([elt[0] for elt in pts]), max([elt[0] for elt in pts])
            s_min, s_max = min([elt[2] for elt in pts]), max([elt[2] for elt in pts])

            results_r[map_value] = [r_min, r_max]
            results_c[map_value] = [c_min, c_max]
            results_s[map_value] = [s_min, s_max]

    # save results to json
    with open(f"inverse_plots/inverse_data/pp_{pp_interval[0]}_{pp_interval[1]}_C.json", 'w', encoding='utf-8') as file:
        json.dump(results_c, file, indent=4)

    with open(f"inverse_plots/inverse_data/pp_{pp_interval[0]}_{pp_interval[1]}_R.json", 'w', encoding='utf-8') as file:
        json.dump(results_r, file, indent=4)

    with open(f"inverse_plots/inverse_data/pp_{pp_interval[0]}_{pp_interval[1]}_S.json", 'w', encoding='utf-8') as file:
        json.dump(results_s, file, indent=4)


    return results_r, results_c, results_s

def plot_inverse_problem_count(param="R"):

    with open(f"inverse_plots/inverse_data/pp_25_35_{param}.json", 'r', encoding='utf-8') as file:
        data1 = json.load(file)

    with open(f"inverse_plots/inverse_data/pp_35_45_{param}.json", 'r', encoding='utf-8') as file:
        data2 = json.load(file)

    with open(f"inverse_plots/inverse_data/pp_45_55_{param}.json", 'r', encoding='utf-8') as file:
        data3 = json.load(file)

    with open(f"inverse_plots/inverse_data/pp_55_65_{param}.json", 'r', encoding='utf-8') as file:
        data4 = json.load(file)

    # # order the data by keys ascending
    # data1 = dict(sorted(data1.items()))
    # data2 = dict(sorted(data2.items()))
    # data3 = dict(sorted(data3.items()))
    # data4 = dict(sorted(data4.items()))

    # print(data1.keys())

    min_param1, max_param1 = [data1[key][0] for key in data1], [data1[key][1] for key in data1]
    min_param2, max_param2 = [data2[key][0] for key in data2], [data2[key][1] for key in data2]
    min_param3, max_param3 = [data3[key][0] for key in data3], [data3[key][1] for key in data3]
    min_param4, max_param4 = [data4[key][0] for key in data4], [data4[key][1] for key in data4]


    fig, axs = plt.subplots(1, 4, figsize=(15, 6))

    # flatten the axes
    axs = axs.flatten()

    axs[0].plot(list(data1.keys()), min_param1, label=f"min {param}")
    axs[0].plot(list(data1.keys()), max_param1, label=f"max {param}")

    axs[1].plot(list(data2.keys()), min_param2, label=f"min {param}")
    axs[1].plot(list(data2.keys()), max_param2, label=f"max {param}")

    axs[2].plot(list(data3.keys()), min_param3, label=f"min {param}")
    axs[2].plot(list(data3.keys()), max_param3, label=f"max {param}")

    axs[3].plot(list(data4.keys()), min_param4, label=f"min {param}")
    axs[3].plot(list(data4.keys()), max_param4, label=f"max {param}")

    # set titles
    axs[0].set_title(f"PP interval: 25 - 35")
    axs[1].set_title(f"PP interval: 35 - 45")
    axs[2].set_title(f"PP interval: 45 - 55")
    axs[3].set_title(f"PP interval: 55 - 65")

    # show every tenth x tick
    for ax in axs.flat:

        ax.set_xticks(list(data1.keys())[::10])
        ax.set_xticklabels(list(data1.keys())[::10])




    plt.tight_layout()
    plt.show()





if __name__ == "__main__":

    df, feature_names = load_new_data()

    # merged_df = load_data_for_residuum_features()
    # # plot_inverse(df, dict())

    # new_merged_df = merged_df[['t_sp', 'max_slope', 'sv', 'bpdn_pp_ratio', 'area_sp_dn']]
    # # print(new_merged_df.describe())

    feature_dict = { # velikostni red 
        't_sp' : 0.02, # 0.02
        'max_slope' : 0.31, # 1
        'bpdn_pp_ratio' : 0.38, # 0.5
        't_start_dn' : 0.01, # 500
        'dbp' : 19.78,
        'map' : 21.6,
        'pp' : 8.81,
    }

    renaming2 = dict((renaming[key], key) for key in renaming)
    fts = [renaming2[feature] for feature in feature_dict] + [renaming2['map'], renaming2['pp']]
    # a = df[fts].describe()
    

    # save datafrane a as json
    # a.to_json("inverse_plots/inverse_plot_test.json")

    feature_dict2 = dict((renaming2[feature], feature_dict[feature]) for feature in feature_dict)

    # res_features = ['max_res', 'min_res', 'min_1st_deriv_res', 'max_1st_deriv_res', 'min_2nd_deriv_res', 'max_2nd_deriv_res', 'area_under_res']

    # reference_pt = {'C': 0.95, 'R': 0.9, 'S': 0.4}

    # # 
    C_values = [0.95, 1.75]
    R_values = [0.9, 1.3]
    S_values = [0.4, 1.0]

    # # Generate all combinations using product from itertools
    reference_pts = [{'C': C, 'R': R, 'S': S} for C, R, S in product(C_values, R_values, S_values)]

    # plot reference pts in 3d scatter with fixed axis limit
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for i in range(len(reference_pts)):
    #     reference_pt = reference_pts[i]
    #     ax.scatter(reference_pt["C"], reference_pt["R"], reference_pt["S"], color='k', s=15)
    #     # add text to point
    #     ax.text(reference_pt["C"], reference_pt["R"], reference_pt["S"], f"{i + 1}", color='k', fontdict={'weight': 'bold', 'size': 15})

    # ax.set_xlabel('kontraktilnost')
    # ax.set_ylabel('upornost')
    # ax.set_zlabel('podajnost')

    # ax.set_xlim((0.55, 2.15))
    # ax.set_ylim((0.7, 1.5))
    # ax.set_zlim((0.1, 1.3))

    # plt.tight_layout()
    # plt.show()    

    # # plot_inverse_single_reference_pt(df, feature_dict, reference_pts[1])

    # feature_dict = { # velikostni red 
    #     't_sp' : 0.02, # 0.02
    #     'max_slope' : 0.31, # 1
    #     'bpdn_pp_ratio' : 0.38, # 0.5
    #     't_start_dn' : 0.01, # 500
    #     'dbp' : 19.78,
    #     'map' : 2.5,
    #     'pp' : 2.5,
    # }
    # feature_dict2 = dict((renaming2[feature], feature_dict[feature]) for feature in feature_dict)
    # multple_reference_plots(df, reference_pts, feature_dict2, filepath="inverse_plots/inverse_plot_test.png")

    # feature_dict = { # velikostni red 
    #     't_sp' : 0.02/2, # 0.02
    #     'max_slope' : 0.31/2, # 1
    #     'bpdn_pp_ratio' : 0.38/2, # 0.5
    #     't_start_dn' : 0.01/2, # 500
    #     'dbp' : 19.78/2,
    #     'map' : 5,
    #     'pp' : 5,
    # }
    # feature_dict2 = dict((renaming2[feature], feature_dict[feature]) for feature in feature_dict)
    # multple_reference_plots(df, reference_pts, feature_dict2, filepath="inverse_plots/inverse_plot_test.png")

    # feature_dict = { # velikostni red 
    #     't_sp' : 0.02/3, # 0.02
    #     'max_slope' : 0.31/3, # 1
    #     'bpdn_pp_ratio' : 0.38/3, # 0.5
    #     't_start_dn' : 0.01/3, # 500
    #     'dbp' : 19.78/3,
    #     'map' : 5,
    #     'pp' : 5,
    # }
    # feature_dict2 = dict((renaming2[feature], feature_dict[feature]) for feature in feature_dict)
    # multple_reference_plots(df, reference_pts, feature_dict2, filepath="inverse_plots/inverse_plot_test.png")


    feature_dict = { # velikostni red 
        't_sp' : 0.02/4, # 0.02
        'max_slope' : 0.31/4, # 1
        'bpdn_pp_ratio' : 0.38/4, # 0.5
        't_start_dn' : 0.01/4, # 500
        'dbp' : 19.78/4,
        'map' : 2.5,
        'pp' : 2.5,
    }
    feature_dict2 = dict((renaming2[feature], feature_dict[feature]) for feature in feature_dict)
    multple_reference_plots(df, reference_pts, feature_dict2, filepath="inverse_plots/inverse_plot_test.png")

    # feature_dict = { # velikostni red 
    #     't_sp' : 0.02, # 0.02
    #     # 'max_slope' : 0.31/4, # 1
    #     # 'bpdn_pp_ratio' : 0.38/4, # 0.5
    #     # 't_start_dn' : 0.01/4, # 500
    #     # 'dbp' : 19.78/4,
    #     'map' : 5,
    #     'pp' : 5,
    # }
    # feature_dict2 = dict((renaming2[feature], feature_dict[feature]) for feature in feature_dict)
    # multple_reference_plots(df, reference_pts, feature_dict2, filepath="inverse_plots/inverse_plot_test.png")

    # feature_dict = { # velikostni red 
    #     # 't_sp' : 0.02/4, # 0.02
    #     'max_slope' : 0.31/4, # 1
    #     # 'bpdn_pp_ratio' : 0.38/4, # 0.5
    #     # 't_start_dn' : 0.01/4, # 500
    #     # 'dbp' : 19.78/4,
    #     'map' : 5,
    #     'pp' : 5,
    # }
    # feature_dict2 = dict((renaming2[feature], feature_dict[feature]) for feature in feature_dict)
    # multple_reference_plots(df, reference_pts, feature_dict2, filepath="inverse_plots/inverse_plot_test.png")

    # feature_dict = { # velikostni red 
    #     # 't_sp' : 0.02/4, # 0.02
    #     # 'max_slope' : 0.31/4, # 1
    #     'bpdn_pp_ratio' : 0.38/4, # 0.5
    #     # 't_start_dn' : 0.01/4, # 500
    #     # 'dbp' : 19.78/4,
    #     'map' : 5,
    #     'pp' : 5,
    # }
    # feature_dict2 = dict((renaming2[feature], feature_dict[feature]) for feature in feature_dict)
    # multple_reference_plots(df, reference_pts, feature_dict2, filepath="inverse_plots/inverse_plot_test.png")

    # feature_dict = { # velikostni red 
    #     # 't_sp' : 0.02/4, # 0.02
    #     # 'max_slope' : 0.31/4, # 1
    #     # 'bpdn_pp_ratio' : 0.38/4, # 0.5
    #     't_start_dn' : 0.01/4, # 500
    #     # 'dbp' : 19.78/4,
    #     'map' : 5,
    #     'pp' : 5,
    # }
    # feature_dict2 = dict((renaming2[feature], feature_dict[feature]) for feature in feature_dict)
    # multple_reference_plots(df, reference_pts, feature_dict2, filepath="inverse_plots/inverse_plot_test.png")

    # feature_dict = { # velikostni red 
    #     # 't_sp' : 0.02/4, # 0.02
    #     # 'max_slope' : 0.31/4, # 1
    #     # 'bpdn_pp_ratio' : 0.38/4, # 0.5
    #     # 't_start_dn' : 0.01/4, # 500
    #     'dbp' : 19.78/4,
    #     'map' : 5,
    #     'pp' : 5,
    # }
    # feature_dict2 = dict((renaming2[feature], feature_dict[feature]) for feature in feature_dict)
    # multple_reference_plots(df, reference_pts, feature_dict2, filepath="inverse_plots/inverse_plot_test.png")

    feature_dict = { # velikostni red 
        't_sp' : [0.02, 0.02/2, 0.02/4], # 0.02
        'max_slope' : [0.31, 0.31/2, 0.31/4], # 1
        'bpdn_pp_ratio' : [0.38, 0.38/2, 0.38/4], # 0.5
        't_start_dn' : [0.01, 0.01/2, 0.01/4], # 500
        'dbp' : [19.78, 19.78/2, 19.78/4],
        # 'map' : 5,
        # 'pp' : 5,
    }

    # parts = read_parsed_data()

    # plot_inverse_3d_and_2d(merged_df, parts, feature_dict, reference_pts)

    # df = color_pts_by_nearby_pts(merged_df, feature_dict)

    # maps, pps = compute_all_maps_and_pps(read_parsed_data())

    # # print(maps[60])

    # inverse_problem_count(maps, pps, pp_interval=[25, 35])
    # inverse_problem_count(maps, pps, pp_interval=[35, 45])
    # inverse_problem_count(maps, pps, pp_interval=[45, 55])
    # inverse_problem_count(maps, pps, pp_interval=[55, 65])

    # plot_inverse_problem_count(param="R")
    # plot_inverse_problem_count(param="C")
    # plot_inverse_problem_count(param="S")





















    # multple_reference_plots(merged_df, reference_pts, feature_dict, filepath="inverse_plots/inverse_plot_max_res.png")

    # ranges: 
    # C : 0.55 - 2.15
    # R : 0.7 - 1.5
    # S :  0.1 - 1.3

    # feature_dict = { # velikostni red 
    #     't_sp' : 0.05, # 0.02
    #     'max_slope' : 1, # 1
    #     'sv' : 10, # 10
    #     'bpdn_pp_ratio' : 0.5, # 0.5
    #     'area_sp_dn' : 5000, # 500

    #     # residuum
    #     # 'max_res' : 25, 
    #     'min_res' : 25, 
    #     # 'min_1st_deriv_res' : 0.1, 
    #     # 'max_1st_deriv_res' : 1, 
    #     # 'min_2nd_deriv_res' : 1, 
    #     # 'max_2nd_deriv_res' : 0.1, 
    #     # 'area_under_res' : 10000 
    # }

    # multple_reference_plots(merged_df, reference_pts, feature_dict, filepath="inverse_plots/inverse_plot_min_res.png")

    # feature_dict = { # velikostni red
    #     't_sp' : 0.05, # 0.02
    #     'max_slope' : 1, # 1
    #     'sv' : 10, # 10
    #     'bpdn_pp_ratio' : 0.5, # 0.5
    #     'area_sp_dn' : 5000, # 500

    #     # residuum
    #     # 'max_res' : 25, 
    #     # 'min_res' : 25, 
    #     'min_1st_deriv_res' : 0.1, 
    #     # 'max_1st_deriv_res' : 1, 
    #     # 'min_2nd_deriv_res' : 1, 
    #     # 'max_2nd_deriv_res' : 0.1, 
    #     # 'area_under_res' : 10000 
    # }

    # multple_reference_plots(merged_df, reference_pts, feature_dict, filepath="inverse_plots/inverse_plot_min_1st_deriv_res.png")

    # feature_dict = { # velikostni red
    #     't_sp' : 0.05, # 0.02
    #     'max_slope' : 1, # 1
    #     'sv' : 10, # 10
    #     'bpdn_pp_ratio' : 0.5, # 0.5
    #     'area_sp_dn' : 5000, # 500

    #     # residuum
    #     # 'max_res' : 25, 
    #     # 'min_res' : 25, 
    #     # 'min_1st_deriv_res' : 0.1, 
    #     'max_1st_deriv_res' : 1, 
    #     # 'min_2nd_deriv_res' : 1, 
    #     # 'max_2nd_deriv_res' : 0.1, 
    #     # 'area_under_res' : 10000 
    # }

    # multple_reference_plots(merged_df, reference_pts, feature_dict, filepath="inverse_plots/inverse_plot_max_1st_deriv_res.png")

    # feature_dict = { # velikostni red
    #     't_sp' : 0.05, # 0.02
    #     'max_slope' : 1, # 1
    #     'sv' : 10, # 10
    #     'bpdn_pp_ratio' : 0.5, # 0.5
    #     'area_sp_dn' : 5000, # 500

    #     # residuum
    #     # 'max_res' : 25, 
    #     # 'min_res' : 25, 
    #     # 'min_1st_deriv_res' : 0.1, 
    #     # 'max_1st_deriv_res' : 1, 
    #     'min_2nd_deriv_res' : 1, 
    #     # 'max_2nd_deriv_res' : 0.1, 
    #     # 'area_under_res' : 10000 
    # }

    # multple_reference_plots(merged_df, reference_pts, feature_dict, filepath="inverse_plots/inverse_plot_min_2nd_deriv_res.png")

    # feature_dict = { # velikostni red
    #     't_sp' : 0.05, # 0.02
    #     'max_slope' : 1, # 1
    #     'sv' : 10, # 10
    #     'bpdn_pp_ratio' : 0.5, # 0.5
    #     'area_sp_dn' : 5000, # 500

    #     # residuum
    #     # 'max_res' : 25, 
    #     # 'min_res' : 25, 
    #     # 'min_1st_deriv_res' : 0.1, 
    #     # 'max_1st_deriv_res' : 1, 
    #     # 'min_2nd_deriv_res' : 1, 
    #     'max_2nd_deriv_res' : 0.1, 
    #     # 'area_under_res' : 10000 
    # }

    # multple_reference_plots(merged_df, reference_pts, feature_dict, filepath="inverse_plots/inverse_plot_max_2nd_deriv_res.png")

    # feature_dict = { # velikostni red
    #     't_sp' : 0.05, # 0.02
    #     'max_slope' : 1, # 1
    #     'sv' : 10, # 10
    #     'bpdn_pp_ratio' : 0.5, # 0.5
    #     'area_sp_dn' : 5000, # 500

    #     # residuum
    #     # 'max_res' : 25, 
    #     # 'min_res' : 25, 
    #     # 'min_1st_deriv_res' : 0.1, 
    #     # 'max_1st_deriv_res' : 1, 
    #     # 'min_2nd_deriv_res' : 1, 
    #     # 'max_2nd_deriv_res' : 0.1, 
    #     'area_under_res' : 10000 
    # }

    # multple_reference_plots(merged_df, reference_pts, feature_dict, filepath="inverse_plots/inverse_plot_area_under_res.png")

    # feature_dict = { # velikostni red 
    #     't_sp' : 0.05, # 0.02
    #     'max_slope' : 1, # 1
    #     'sv' : 10, # 10
    #     'bpdn_pp_ratio' : 0.5, # 0.5
    #     'area_sp_dn' : 5000, # 500

    #     # residuum
    #     'max_res' : 2.5, 
    #     # 'min_res' : 25, 
    #     # 'min_1st_deriv_res' : 0.1, 
    #     # 'max_1st_deriv_res' : 1, 
    #     # 'min_2nd_deriv_res' : 1, 
    #     # 'max_2nd_deriv_res' : 0.1, 
    #     # 'area_under_res' : 10000 
    # }

    # multple_reference_plots(merged_df, reference_pts, feature_dict, filepath="inverse_plots/inverse_plot_max_res_2.png")

    # feature_dict = { # velikostni red 
    #     't_sp' : 0.05, # 0.02
    #     'max_slope' : 1, # 1
    #     'sv' : 10, # 10
    #     'bpdn_pp_ratio' : 0.5, # 0.5
    #     'area_sp_dn' : 5000, # 500

    #     # residuum
    #     # 'max_res' : 25, 
    #     'min_res' : 2.5, 
    #     # 'min_1st_deriv_res' : 0.1, 
    #     # 'max_1st_deriv_res' : 1, 
    #     # 'min_2nd_deriv_res' : 1, 
    #     # 'max_2nd_deriv_res' : 0.1, 
    #     # 'area_under_res' : 10000 
    # }

    # multple_reference_plots(merged_df, reference_pts, feature_dict, filepath="inverse_plots/inverse_plot_min_res_2.png")

    # feature_dict = { # velikostni red
    #     't_sp' : 0.05, # 0.02
    #     'max_slope' : 1, # 1
    #     'sv' : 10, # 10
    #     'bpdn_pp_ratio' : 0.5, # 0.5
    #     'area_sp_dn' : 5000, # 500

    #     # residuum
    #     # 'max_res' : 25, 
    #     # 'min_res' : 25, 
    #     'min_1st_deriv_res' : 0.01, 
    #     # 'max_1st_deriv_res' : 1, 
    #     # 'min_2nd_deriv_res' : 1, 
    #     # 'max_2nd_deriv_res' : 0.1, 
    #     # 'area_under_res' : 10000 
    # }

    # multple_reference_plots(merged_df, reference_pts, feature_dict, filepath="inverse_plots/inverse_plot_min_1st_deriv_res_2.png")

    # feature_dict = { # velikostni red
    #     't_sp' : 0.05, # 0.02
    #     'max_slope' : 1, # 1
    #     'sv' : 10, # 10
    #     'bpdn_pp_ratio' : 0.5, # 0.5
    #     'area_sp_dn' : 5000, # 500

    #     # residuum
    #     # 'max_res' : 25, 
    #     # 'min_res' : 25, 
    #     # 'min_1st_deriv_res' : 0.1, 
    #     'max_1st_deriv_res' : 0.1, 
    #     # 'min_2nd_deriv_res' : 1, 
    #     # 'max_2nd_deriv_res' : 0.1, 
    #     # 'area_under_res' : 10000 
    # }

    # multple_reference_plots(merged_df, reference_pts, feature_dict, filepath="inverse_plots/inverse_plot_max_1st_deriv_res_2.png")

    # feature_dict = { # velikostni red
    #     't_sp' : 0.05, # 0.02
    #     'max_slope' : 1, # 1
    #     'sv' : 10, # 10
    #     'bpdn_pp_ratio' : 0.05, # 0.5
    #     'area_sp_dn' : 5000, # 500

    #     # residuum
    #     # 'max_res' : 25, 
    #     # 'min_res' : 25, 
    #     # 'min_1st_deriv_res' : 0.1, 
    #     # 'max_1st_deriv_res' : 1, 
    #     'min_2nd_deriv_res' : 0.1, 
    #     # 'max_2nd_deriv_res' : 0.1, 
    #     # 'area_under_res' : 10000 
    # }

    # multple_reference_plots(merged_df, reference_pts, feature_dict, filepath="inverse_plots/inverse_plot_min_2nd_deriv_res_2.png")

    # feature_dict = { # velikostni red
    #     't_sp' : 0.05, # 0.02
    #     'max_slope' : 1, # 1
    #     'sv' : 10, # 10
    #     'bpdn_pp_ratio' : 0.5, # 0.5
    #     'area_sp_dn' : 5000, # 500

    #     # residuum
    #     # 'max_res' : 25, 
    #     # 'min_res' : 25, 
    #     # 'min_1st_deriv_res' : 0.1, 
    #     # 'max_1st_deriv_res' : 1, 
    #     # 'min_2nd_deriv_res' : 1, 
    #     'max_2nd_deriv_res' : 0.01, 
    #     # 'area_under_res' : 10000 
    # }

    # multple_reference_plots(merged_df, reference_pts, feature_dict, filepath="inverse_plots/inverse_plot_max_2nd_deriv_res_2.png")

    # feature_dict = { # velikostni red
    #     't_sp' : 0.05, # 0.02
    #     'max_slope' : 1, # 1
    #     'sv' : 10, # 10
    #     'bpdn_pp_ratio' : 0.5, # 0.5
    #     'area_sp_dn' : 5000, # 500

    #     # residuum
    #     # 'max_res' : 25, 
    #     # 'min_res' : 25, 
    #     # 'min_1st_deriv_res' : 0.1, 
    #     # 'max_1st_deriv_res' : 1, 
    #     # 'min_2nd_deriv_res' : 1, 
    #     # 'max_2nd_deriv_res' : 0.1, 
    #     'area_under_res' : 1000 
    # }

    # multple_reference_plots(merged_df, reference_pts, feature_dict, filepath="inverse_plots/inverse_plot_area_under_res_2.png")

