import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, make_scorer, precision_score, log_loss

from lasso_regression import split_dataset

import time
import json

from magistrska_koda.xgboost_implementation_first import load_new_data

from feature_renaming import renaming


def get_feature_scores(params, early_stopping_rounds, num_round, X_train, X_test, y_train, y_test):

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    # num_round = 100  # Number of boosting rounds
    bst = xgb.train(params, dtrain, num_round, evallist, early_stopping_rounds=early_stopping_rounds)

    feature_importance_weight = bst.get_score(importance_type='weight')
    feature_importance_gain = bst.get_score(importance_type='gain')
    feature_importance_cover = bst.get_score(importance_type='cover')

    return feature_importance_weight, feature_importance_gain, feature_importance_cover


def get_feature_scores_cv(params, early_stopping_rounds, num_round, X, C, R, S, title='contractiliy', mode='cv', k=10, compute_r2=False):

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
        # num_round = 100  # Number of boosting rounds
        bst = xgb.train(params, dtrain, num_round, evallist, early_stopping_rounds=early_stopping_rounds)

        if compute_r2:
            y_pred = bst.predict(dtest)
            r2 = r2_score(y_test, y_pred)
            # print(f'R²: {r2:.10f}')
            r2s.append(r2)

        weight, gain, cover = bst.get_score(importance_type='weight'), bst.get_score(importance_type='gain'), bst.get_score(importance_type='cover')

        for key in weight:
            feature_importance_weight[key].append(weight[key])
            feature_importance_gain[key].append(gain[key])
            feature_importance_cover[key].append(cover[key])


    if compute_r2:
        return feature_importance_weight, feature_importance_gain, feature_importance_cover, r2s
    
    else:
        return feature_importance_weight, feature_importance_gain, feature_importance_cover
    

def get_mean_feature_score_cv(params, early_stopping_rounds, num_round, X, C, R, S, title='contractiliy', mode='cv', k=10, save=False, r2s=False):

    feature_importance_weight, feature_importance_gain, feature_importance_cover, r2s = get_feature_scores_cv(params, early_stopping_rounds, num_round, X, C, R, S, title=title, mode=mode, k=k, compute_r2=True)

    
    final_result = dict((feature, [np.mean(feature_importance_weight[feature]) if feature_importance_weight[feature] != [] else 0, 
                                   np.mean(feature_importance_gain[feature]) if feature_importance_gain[feature] != [] else 0,
                                   np.mean(feature_importance_cover[feature]) if feature_importance_cover[feature] != [] else 0
                                   ]) for feature in feature_importance_weight.keys())


    if save: 
        with open(f"xgboost_plots/{title}_feature_importance_20240911.json", 'w', encoding='utf-8') as file:
            json.dump(final_result, file, indent=4)

    if r2s:
        return final_result, r2s
    else:
        return final_result

def feature_importance_bar_plot(filepath, final_result=None, choice=None, plot_title='kontraktilnost', remove=None):

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
    print(len(sorted_features))

    # sorted_features = dict((renaming[feature], sorted_features[feature]) for feature in sorted_features)
    # sorted_features = [renaming[feature] for feature in sorted_features]

    # print(sorted_features)
    # feature_keys = dict((i, feature) for i, feature in enumerate(sorted_features))
    # print(feature_keys)

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plotting all values or just one depending on choice
    for i, feature in enumerate(sorted_features):
        if choice is None:
            if feature in sorted_features:
                ax.barh(i - 0.2, final_result[feature][0], height=0.2, label='Weight' if i == 0 else "", color='b')
                ax.barh(i, final_result[feature][1], height=0.2, label='Gain' if i == 0 else "", color='g')
                ax.barh(i + 0.2, final_result[feature][2], height=0.2, label='Cover' if i == 0 else "", color='r')
        else:
            if feature in sorted_features:
                
                title = 'Weight' if choice == 0 else 'Gain' if choice == 1 else 'Cover'
                ax.barh(i, final_result[feature][choice], height=0.6, label=f'{title}' if i == 0 else "", color='blue')

    # Set feature names as x-axis labels
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features, ha='right')

    # Add labels and title
    ax.set_ylabel('Značilke')
    ax.set_xlabel('Pomembnost značilk')
    ax.set_title(f'Pomembnost značilk za {plot_title}')



    # Show legend if plotting all values
    # if choice is None:
    ax.legend(loc='lower right')

    # Display the plot
    plt.savefig(f'xgboost_plots/{plot_title}_feature_importance_bar_plot_20240911.png')
    # plt.tight_layout()
    plt.show()

def alpha_plots(alphas, X, C, R, S, title='contractility', mode='cv', k=10):

    final_results = {}
    r2s_result = {}

    for alpha in alphas:

        if title == 'contractility':
            p = {
                "objective": "reg:squarederror",
                "max_depth": 8,
                "subsample": 1,
                "eta": 0.05,
                "eval_metric": "rmse",
                "seed": 42,
                'alpha' : alpha
            }
            num_round = 1500
            early_stopping_rounds = 50

        elif title == 'resistance':

            p = {
                "objective": "reg:squarederror",
                "max_depth": 8,
                "subsample": 1,
                "eta": 0.05,
                "eval_metric": "rmse",
                "seed": 42,
                'alpha' : alpha
            }
            num_round = 1500
            early_stopping_rounds = 50

        elif title == 'stiffness':

            p = {
                "objective": "reg:squarederror",
                "max_depth": 8,
                "subsample": 1,
                "eta": 0.05,
                "eval_metric": "rmse",
                "seed": 42,
                'alpha' : alpha
            }
            num_round = 1500
            early_stopping_rounds = 50

        final_result, r2s = get_mean_feature_score_cv(p, early_stopping_rounds, num_round, X, C, R, S, title=title, mode=mode, k=k, save=False, r2s=True)
        # final_result['r2'] = np.mean(r2s)
        final_results[alpha] = final_result
        r2s_result[alpha] = np.mean(r2s)

    # dobili smo za vsak alfa 3 vrednosti feature importancov in r2
    first_key = list(final_results.keys())[0]
    gain, weight, cover, r2 = dict((feature, []) for feature in final_results[first_key].keys()), dict((feature, []) for feature in final_results[first_key].keys()), dict((feature, []) for feature in final_results[first_key].keys()), []

    for alpha in final_results.keys():
        for feature in final_results[alpha].keys():

            # za vsak feature dobimo vse vrednosti za vsak alpha

            gain[feature].append(final_results[alpha][feature][0])
            weight[feature].append(final_results[alpha][feature][1])
            cover[feature].append(final_results[alpha][feature][2])
            # r2.append(final_results[alpha][feature][3])

    # save as json
    with open(f"xgboost_plots/{title}_alpha_results_gain_20240912_2.json", 'w', encoding='utf-8') as file:
        json.dump(gain, file, indent=4)

    with open(f"xgboost_plots/{title}_alpha_results_weight_20240912_2.json", 'w', encoding='utf-8') as file:
        json.dump(weight, file, indent=4)
    
    with open(f"xgboost_plots/{title}_alpha_results_r2_20240912_2.json", 'w', encoding='utf-8') as file:
        json.dump(r2s_result, file, indent=4)

    with open(f"xgboost_plots/{title}_alpha_results_cover_20240912_2.json", 'w', encoding='utf-8') as file:
        json.dump(cover, file, indent=4)
    


    return alphas, gain, weight, cover, r2s_result

# def plot_alpha_results(alphas, gain, weight, cover, r2):

#     fig, ax = plt.subplots(2, 2, figsize=(10, 10))

#     ax[0, 0].plot(alphas, [np.mean(values) for values in gain.values()], label='Gain')    

        # naredim CV, zračunam R2 in vrednosti za featurje
def load_results():
    with open(f"xgboost_plots/contractility_alpha_results_gain_20240912_2.json", 'r', encoding='utf-8') as file:
        contractility_gain = json.load(file)

    with open(f"xgboost_plots/contractility_alpha_results_weight_20240912_2.json", 'r', encoding='utf-8') as file:
        contractility_weight = json.load(file)

    with open(f"xgboost_plots/contractility_alpha_results_cover_20240912_2.json", 'r', encoding='utf-8') as file:
        contractility_cover = json.load(file)

    with open(f"xgboost_plots/contractility_alpha_results_r2_20240912_2.json", 'r', encoding='utf-8') as file:
        contractility_r2 = json.load(file)

    with open(f"xgboost_plots/resistance_alpha_results_gain_20240912_2.json", 'r', encoding='utf-8') as file:
        resistance_gain = json.load(file)

    with open(f"xgboost_plots/resistance_alpha_results_weight_20240912_2.json", 'r', encoding='utf-8') as file:
        resistance_weight = json.load(file)

    with open(f"xgboost_plots/resistance_alpha_results_cover_20240912_2.json", 'r', encoding='utf-8') as file:
        resistance_cover = json.load(file)

    with open(f"xgboost_plots/resistance_alpha_results_r2_20240912_2.json", 'r', encoding='utf-8') as file:
        resistance_r2 = json.load(file)

    with open(f"xgboost_plots/stiffness_alpha_results_gain_20240912_2.json", 'r', encoding='utf-8') as file:
        stiffness_gain = json.load(file)

    with open(f"xgboost_plots/stiffness_alpha_results_weight_20240912_2.json", 'r', encoding='utf-8') as file:
        stiffness_weight = json.load(file)

    with open(f"xgboost_plots/stiffness_alpha_results_cover_20240912_2.json", 'r', encoding='utf-8') as file:
        stiffness_cover = json.load(file)

    with open(f"xgboost_plots/stiffness_alpha_results_r2_20240912_2.json", 'r', encoding='utf-8') as file:
        stiffness_r2 = json.load(file)

    return contractility_gain, contractility_weight, contractility_cover, contractility_r2, resistance_gain, resistance_weight, resistance_cover, resistance_r2, stiffness_gain, stiffness_weight, stiffness_cover, stiffness_r2

def plot_alpha_results(feature_scores, r2, title='kontraktilnost', importance_type='gain', show_r2=False, top_n=None):

    # plot one plot where we take the keys of r2 as x axis. 
    # then we plot the value od r2 in black as a line
    # and then we plot the lists for each key in feature_scores as a line on the same graph 
    # we have different y axes for r2 (from 0 to 1) and for feature_scores  (from min to max value of all values in all lists)

    max_values = {feature: max([abs(score) for score in scores]) for feature, scores in feature_scores.items()}

    # # Sort features by maximum value in descending order and select the top N
    # if top_n is not None:
    #     feature_scores = sorted(max_values, key=max_values.get, reverse=True)[:top_n]

    fig, ax = plt.subplots(figsize=(13, 10))
    plt.gca().set_xscale('log')

    x_axis = list(r2.keys())
    x_axis = [float(elt) for elt in x_axis]


    if top_n is not None:
        sorted_features = sorted(max_values, key=max_values.get, reverse=True)[:top_n]
        for feature in sorted_features:
            ax.plot(list(range(len(feature_scores[feature]))), feature_scores[feature], label=renaming[feature])

    else: 
        # for feature in feature_scores.keys():
        for feature in feature_scores.keys():
            ax.plot(x_axis, feature_scores[feature], label=renaming[feature])



    ax.set_xlabel('Vrednosti regularizacijskega parametra α')
    ax.set_ylabel('Pomembnost značilk')

    tick_positions = x_axis[::5] + [x_axis[-1]]

    # Convert tick positions to the form 10^__
    tick_labels = [f'$10^{{{round(tick, 4)}}}$' for tick in tick_positions]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    legend = ax.legend(bbox_to_anchor = (1, 1), ncol=1)

    if show_r2:
        ax2 = ax.twinx()
        ax2.plot(x_axis, list(r2.values()), color='black', label='R²', linewidth=2)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('R²')
        ax2.legend()

    # title
    if show_r2:
        if top_n:
            ax.set_title(f'Pomembnost značilk ({importance_type}) in R² za {title}  \n v odvisnosti od regularizacijskega parametra α \n za top {top_n} značilk')
        else: 
            ax.set_title(f'Pomembnost značilk ({importance_type}) in R² za {title} \n v odvisnosti od regularizacijskega parametra α')
    else:
        if top_n:
            ax.set_title(f'Pomembnost značilk ({importance_type}) za {title} \n v odvisnosti od regularizacijskega parametra α \n za top {top_n} značilk')
        else: 
            ax.set_title(f'Pomembnost značilk ({importance_type}) za {title} \n v odvisnosti od regularizacijskega parametra α')
    plt.tight_layout()
    plt.savefig(f'xgboost_plots/{title}_{importance_type}_alpha_plot_20240912.png')
    
    plt.show()

def plot_r2(contractility_r2, resistance_r2, stiffness_r2):

    # plot one plot where we take the keys of r2 as x axis.
    # then plot the value odf all three given dictionaries on the same plot with different colors

    fig, ax = plt.subplots()
    plt.gca().set_xscale('log')

    x_axis = list(contractility_r2.keys())
    x_axis = [float(elt) for elt in x_axis]

    ax.plot(x_axis, list(contractility_r2.values()), label='Kontraktilnost', color='blue')
    ax.plot(x_axis, list(resistance_r2.values()), label='Upornost', color='red')
    ax.plot(x_axis, list(stiffness_r2.values()), label='Podajnost', color='green')

    ax.set_xlabel('Vrednosti regularizacijskega parametra α')
    ax.set_ylabel('R²')

    tick_positions = x_axis[::5] + [x_axis[-1]]

    # Convert tick positions to the form 10^__
    tick_labels = [f'$10^{{{round(tick, 4)}}}$' for tick in tick_positions]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.legend()

    ax.set_title('R² za kontraktilnost, upornost in podajnost \n v odvisnosti od regularizacijskega parametra α')

    plt.tight_layout()
    plt.savefig('xgboost_plots/r2_plot_20240912.png')

    
    plt.show()


if __name__=="__main__":

    X, C, R, S, feature_names = load_new_data()
    # splits = split_dataset(X, C, R, S, mode='cv', k=10)
    # split = splits[0]

    # # X_train, X_test, y_train, y_test = split[0], split[1], split[2], split[3]

    # # feature_importance_weight, feature_importance_gain, feature_importance_cover = get_feature_scores(params, early_stopping_rounds, num_round, X_train, X_test, y_train, y_test)
    # feature_importance_weight, feature_importance_gain, feature_importance_cover = get_feature_scores_cv(params, early_stopping_rounds, num_round, X, C, R, S, title='contractility', mode='cv', k=10)

    # final_result = dict((feature, [np.mean(feature_importance_weight[feature]), 
    #                                np.mean(feature_importance_gain[feature]),
    #                                np.mean(feature_importance_cover[feature])]) for feature in feature_importance_weight.keys())

    #==========================================================

    # p = {
    #     "objective": "reg:squarederror",
    #     "max_depth": 8,
    #     "subsample": 1,
    #     "eta": 0.05,
    #     "eval_metric": "rmse",
    #     "seed": 42
    # }
    # num_round = 1500
    # early_stopping_rounds = 50

    # final_result = get_mean_feature_score_cv(p, early_stopping_rounds, num_round, X, C, R, S, title='contractility', mode='cv', k=10, save=True)
    # feature_importance_bar_plot(filepath="xgboost_plots/contractility_feature_importance_20240911.json", final_result=final_result, choice=1, plot_title='kontraktilnost')


    # p = {
    #     "objective": "reg:squarederror",
    #     "max_depth": 8,
    #     "subsample": 1,
    #     "eta": 0.05,
    #     "eval_metric": "rmse",
    #     "seed": 42
    # }
    # num_round = 1500
    # early_stopping_rounds = 50

    # final_result = get_mean_feature_score_cv(p, early_stopping_rounds, num_round, X, C, R, S, title='resistance', mode='cv', k=10, save=True)
    # feature_importance_bar_plot(filepath="xgboost_plots/resistance_feature_importance_20240911.json", final_result=final_result, choice=1, plot_title='upornost')


    # p = {
    #     "objective": "reg:squarederror",
    #     "max_depth": 8,
    #     "subsample": 1,
    #     "eta": 0.05,
    #     "eval_metric": "rmse",
    #     "seed": 42
    # }
    # num_round = 1500
    # early_stopping_rounds = 50

    # final_result = get_mean_feature_score_cv(p, early_stopping_rounds, num_round, X, C, R, S, title='stiffness', mode='cv', k=10, save=True)
    # feature_importance_bar_plot(filepath="xgboost_plots/stiffness_feature_importance_20240911.json", final_result=final_result, choice=1, plot_title='podajnost')


    #==========================================================

    # final_result = get_mean_feature_score_cv(params, early_stopping_rounds, num_round, X, C, R, S, title='contractility', mode='cv', k=10)
    # final_result = get_mean_feature_score_cv(params, early_stopping_rounds, num_round, X, C, R, S, title='stiffness', mode='cv', k=10)
    # final_result = get_mean_feature_score_cv(params, early_stopping_rounds, num_round, X, C, R, S, title='resistance', mode='cv', k=10)

    # feature_importance_bar_plot(filepath="xgboost_plots/contractility_feature_importance_20240911.json", choice=1, plot_title='kontraktilnost', remove=2)
    # feature_importance_bar_plot(filepath="xgboost_plots/resistance_feature_importance_20240911.json", choice=1, plot_title='upornost', remove=1)
    # feature_importance_bar_plot(filepath="xgboost_plots/stiffness_feature_importance_20240911.json", choice=1, plot_title='podajnost')

    # X, C, R, S, feature_names = load_new_data()


    #==========================================================

    # alphas = np.logspace(-3, 0.5, 10)
    # # # # # alphas = [1, 10]

    # a, gain, weight, cover, r2 = alpha_plots(alphas, X, C, R, S, title='contractility', mode='cv', k=10)
    # a, gain, weight, cover, r2 = alpha_plots(alphas, X, C, R, S, title='stiffness', mode='cv', k=10)
    # a, gain, weight, cover, r2 = alpha_plots(alphas, X, C, R, S, title='resistance', mode='cv', k=10)

    contractility_gain, contractility_weight, contractility_cover, contractility_r2, resistance_gain, resistance_weight, resistance_cover, resistance_r2, stiffness_gain, stiffness_weight, stiffness_cover, stiffness_r2 = load_results()

    plot_alpha_results(contractility_gain, contractility_r2, title='kontraktilnost',importance_type='gain', top_n=10)
    plot_alpha_results(resistance_gain, contractility_r2, title='upornost', importance_type='gain',top_n=10)
    plot_alpha_results(stiffness_gain, contractility_r2, title='podajnost', importance_type='gain',top_n=10)

    #================================================================================================

    # plot_alpha_results(contractility_weight, contractility_r2, title='kontraktilnost',importance_type='weight', top_n=20)
    # plot_alpha_results(resistance_weight, contractility_r2, title='upornost', importance_type='weight',top_n=20)
    # plot_alpha_results(stiffness_weight, contractility_r2, title='podajnost',importance_type='weight', top_n=20)

    # plot_alpha_results(contractility_cover, contractility_r2, title='kontraktilnost',importance_type='cover', top_n=20)
    # plot_alpha_results(resistance_cover, contractility_r2, title='upornost', importance_type='cover',top_n=20)
    # plot_alpha_results(stiffness_cover, contractility_r2, title='podajnost',importance_type='cover', top_n=20)

    # plot_r2(contractility_r2, resistance_r2, stiffness_r2)

