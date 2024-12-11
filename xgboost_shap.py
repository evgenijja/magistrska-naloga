import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json

import shap


import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, make_scorer, precision_score, log_loss

from lasso_regression_implementation import split_dataset
from feature_renaming import renaming

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

    return X, C, R, S, feature_names

def shap_cross_validation(X_train, X_test, y_train, y_test, filepath1, filepath2):

    Xd = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    evallist = [(Xd, 'train'), (dtest, 'eval')]
    num_round=100
    model = xgb.train({}, Xd, num_round, evallist)

    explainer = shap.TreeExplainer(model)
    # print("explainer done")
    explanation = explainer(Xd)

    # print("shap values")
    # shap_values = explanation.values

    shap_interaction_values = explainer.shap_interaction_values(dtest)
    shap_interaction_values_list = [shap_interaction_value.tolist() for shap_interaction_value in shap_interaction_values]

    # save to jsonž
    with open(filepath1, 'w') as f:
        json.dump(shap_interaction_values_list, f)

    shap_values = explainer.shap_values(dtest)
    shap_values_list = [shap_value.tolist() for shap_value in shap_values]

    # save to json
    with open(filepath2, 'w') as f:
        json.dump(shap_values_list, f)


def load_results(filepaths):

    all_data = []

    for filepath in filepaths:

        with open(filepath, 'r') as f:
            data = json.load(f)

        for d in data:
            all_data.append(d)

    return all_data

def plot_heatmap(all_data):
    """
    all data is a list of matrices (list of lists)
    we want to create an "average matrix" from all the matrices
    """
    
    all_data = np.array(all_data)

    average_matrix = np.mean(all_data, axis=0)

    plt.imshow(average_matrix, cmap='viridis')
    plt.colorbar()
    plt.show()

    return average_matrix


if __name__=="__main__":

    X, C, R, S, feature_names = load_new_data()

    splits = split_dataset(X, C, R, S, mode='cv', k=10)

    # for i, split in enumerate(splits):

    #     print(f"Split {i}")

    #     X_train, X_test, C_train, C_test, R_train, R_test, S_train, S_test = split[0], split[1], split[2], split[3], split[4], split[5], split[6], split[7]

    #     print("C")
    #     shap_cross_validation(X_train, X_test, C_train, C_test, f'shap/shap_interaction_values_C_{i}.json', f'data/shap_values_C_{i}.json')
    #     print("R")
    #     shap_cross_validation(X_train, X_test, R_train, R_test, f'shap/shap_interaction_values_R_{i}.json', f'data/shap_values_R_{i}.json')
    #     print("S")
    #     shap_cross_validation(X_train, X_test, S_train, S_test, f'shap/shap_interaction_values_S_{i}.json', f'data/shap_values_S_{i}.json')

    # C_filepaths = [
    #     "shap/shap_interaction_values_C_0.json",
    #     "shap/shap_interaction_values_C_1.json",
    #     "shap/shap_interaction_values_C_2.json",
    #     "shap/shap_interaction_values_C_3.json",
    #     "shap/shap_interaction_values_C_4.json",
    #     "shap/shap_interaction_values_C_5.json",
    #     "shap/shap_interaction_values_C_6.json",
    #     "shap/shap_interaction_values_C_7.json",
    #     "shap/shap_interaction_values_C_8.json",
    #     "shap/shap_interaction_values_C_9.json"
    # ]

    # all_data = load_results(C_filepaths)
    # plot_heatmap(all_data)

    # R_filepaths = [
    #     "shap/shap_interaction_values_R_0.json",
    #     "shap/shap_interaction_values_R_1.json",
    #     "shap/shap_interaction_values_R_2.json",
    #     "shap/shap_interaction_values_R_3.json",
    #     "shap/shap_interaction_values_R_4.json",
    #     "shap/shap_interaction_values_R_5.json",
    #     "shap/shap_interaction_values_R_6.json",
    #     "shap/shap_interaction_values_R_7.json",
    #     "shap/shap_interaction_values_R_8.json",
    #     "shap/shap_interaction_values_R_9.json"
    # ]

    # all_data = load_results(R_filepaths)
    # plot_heatmap(all_data)

    S_filepaths = [
        "shap/shap_interaction_values_S_0.json",
        "shap/shap_interaction_values_S_1.json",
        "shap/shap_interaction_values_S_2.json",
        "shap/shap_interaction_values_S_3.json",
        "shap/shap_interaction_values_S_4.json",
        "shap/shap_interaction_values_S_5.json",
        "shap/shap_interaction_values_S_6.json",
        "shap/shap_interaction_values_S_7.json",
        "shap/shap_interaction_values_S_8.json",
        "shap/shap_interaction_values_S_9.json"
    ]

    all_data = load_results(S_filepaths)
    average_matrix = plot_heatmap(all_data)

    res = []
    res_diag = []

    res_dict = dict()
    res_diag_dict = dict()

    print(average_matrix.shape)

    names = dict()

    for i in range(average_matrix.shape[0]):
        for j in range(average_matrix.shape[1]):
            if i != j:
                if abs(average_matrix[i][j]) > 0.01 and [j, i] not in res:
                    res.append([i, j])

                    names[i] = feature_names[i]
                    names[j] = feature_names[j]

                    res_dict[(i, j)] = average_matrix[i][j]
            else:
                if abs(average_matrix[i][j]) > 0.01:
                    res_diag.append(i)

                    names[i] = feature_names[i]

                    res_diag_dict[i] = average_matrix[i][j]

    print(res)
    print(res_diag)

    print(res_dict)
    print(res_diag_dict)

    # order res_diag_dict by absolute values
    res_diag_dict = dict(sorted(res_diag_dict.items(), key=lambda item: abs(item[1]), reverse=True))
    

    print(names)

