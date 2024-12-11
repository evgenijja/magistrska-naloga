import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json


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


def get_feature_scores_cv(X, C, R, S, title='contractiliy', mode='cv', k=10, compute_r2=False):

    splits = split_dataset(X, C, R, S, mode=mode, k=k)


    features = X.columns.tolist()

    feature_importance_weight, feature_importance_gain, feature_importance_cover = dict((feature, []) for feature in features), dict((feature, []) for feature in features), dict((feature, []) for feature in features)
    r2s = []

    for split in splits:

        if title == 'contractility':
            X_train, X_test, y_train, y_test = split[0], split[1], split[2], split[3]
        elif title == 'resistance':
            X_train, X_test, y_train, y_test = split[0], split[1], split[4], split[5]
        elif title == 'stiffness':
            X_train, X_test, y_train, y_test = split[0], split[1], split[6], split[7]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        evallist = [(dtrain, 'train'), (dtest, 'eval')]
        num_round = 100  # Number of boosting rounds
        bst = xgb.train(
            {},
            dtrain, num_round, evallist
        )

        if compute_r2:
            y_pred = bst.predict(dtest)
            r2 = r2_score(y_test, y_pred)
            # print(f'R²: {r2:.10f}')
            r2s.append(r2)

        weight, gain, cover = bst.get_score(importance_type='weight'), bst.get_score(importance_type='gain'), bst.get_score(importance_type='cover')

        for key in weight:
            # za vsak feature dobimo seznam vrednosti pri vsakem foldu
            feature_importance_weight[key].append(weight[key])
            feature_importance_gain[key].append(gain[key])
            feature_importance_cover[key].append(cover[key])


    if compute_r2:
        return feature_importance_weight, feature_importance_gain, feature_importance_cover, r2s
    
    else:
        return feature_importance_weight, feature_importance_gain, feature_importance_cover
    

def get_mean_feature_score_cv(X, C, R, S, title='contractiliy', mode='cv', k=10, save=False, r2s=False):

    feature_importance_weight, feature_importance_gain, feature_importance_cover, r2s = get_feature_scores_cv(X, C, R, S, title=title, mode=mode, k=k, compute_r2=True)

    
    final_result = dict((feature, [np.mean(feature_importance_weight[feature]) if feature_importance_weight[feature] != [] else 0, 
                                   np.mean(feature_importance_gain[feature]) if feature_importance_gain[feature] != [] else 0,
                                   np.mean(feature_importance_cover[feature]) if feature_importance_cover[feature] != [] else 0
                                   ]) for feature in feature_importance_weight.keys())


    if save: 
        with open(f"xgboost_final/{title}_feature_importance_FINAL.json", 'w', encoding='utf-8') as file:
            json.dump(final_result, file, indent=4)

    if r2s:
        return final_result, r2s
    else:
        return final_result


def print_feature_importances(filepath, final_result=None, choice=None, plot_title='kontraktilnost', remove=None):

    if final_result is None:
        with open(filepath, 'r', encoding='utf-8') as file:
            final_result = json.load(file)

    # if final_result is None:
    final_result = dict((renaming[feature], values) for feature, values in final_result.items())


    means = {feature: np.mean(values) if values != [None, None, None] else 0 for feature, values in final_result.items()}

    # Sort f    eatures by their mean values
    if choice is None:
        sorted_features = sorted(means, key=means.get)
    else:
        # sort by values[choice]
        sorted_features = sorted(final_result, key=lambda x: final_result[x][choice])
    # print(sorted_features)

    # remove features where final_result[x][choice] = 0
    sorted_features = [feature for feature in sorted_features if round(final_result[feature][choice], 2) != 0]
    print(len(sorted_features))
    if remove is not None:
        # remove top N features
        sorted_features = sorted_features[:len(sorted_features)-remove]

    final_dict = {}
    
    

    for i, feature in enumerate(sorted_features):

        if feature in sorted_features:
            
            # title = 'Weight' if choice == 0 else 'Gain' if choice == 1 else 'Cover'
            final_dict[feature] = final_result[feature][choice]

    # order dict by values in descending order
    final_dict = dict(sorted(final_dict.items(), key=lambda item: item[1], reverse=True))
        

    # save sorted features to json
    with open(f"xgboost_final/{plot_title}_xgboost_result.json", 'w', encoding='utf-8') as file:
        json.dump(final_dict, file, indent=4)

    return sorted_features

if __name__ == "__main__":

    X, C, R, S, feature_names = load_new_data()
    splits = split_dataset(X, C, R, S, mode='cv', k=10)
    
    final_result, r2s_contractility = get_mean_feature_score_cv(X, C, R, S, title='contractility', mode='cv', k=10, save=True, r2s=True)
    sorted_features = print_feature_importances(filepath="xgboost_final/contractility_feature_importance_FINAL.json", final_result=final_result, choice=1, plot_title='kontraktilnost')
    

    final_result, r2s_resistance = get_mean_feature_score_cv(X, C, R, S, title='resistance', mode='cv', k=10, save=True, r2s=True)
    sorted_features = print_feature_importances(filepath="xgboost_final/resistance_feature_importance_FINAL.json", final_result=final_result, choice=1, plot_title='upornost')
    

    final_result, r2s_stiffness = get_mean_feature_score_cv(X, C, R, S, title='stiffness', mode='cv', k=10, save=True, r2s=True)
    sorted_features = print_feature_importances(filepath="xgboost_final/stiffness_feature_importance_FINAL.json", final_result=final_result, choice=1, plot_title='togost')
    
    print(f"Average R² for contractility: {np.mean(r2s_contractility):.10f}")
    print(f"Average R² for resistance: {np.mean(r2s_resistance):.10f}")
    print(f"Average R² for stiffness: {np.mean(r2s_stiffness):.10f}")



