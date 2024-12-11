import numpy as np
import pandas as pd
import json

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, make_scorer, precision_score, log_loss

# from feature_definition.feature_definition_framework import FEATURES

from feature_renaming import renaming

import matplotlib.pyplot as plt

from lasso_regression import lasso_regression, parse_coeffs, split_dataset, load_new_data

"""
SCRIPT FOR 10 FOLD CROSS VALIDATION
"""

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

X, C, R, S, feature_names = load_new_data()

# hočemo meti fiksne barve za featurje
colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
feature_colors = dict((feature, colors[i]) for i, feature in enumerate(feature_names))

def make_single_plot(coeffs, alphas, title, fold_number, colors=feature_colors):
    """
    coeffs ... slovar vrednosti koeficientov pri različnih alpha vrednostih za vse značilke
    alphas ... seznam vseh alpha vrednosti
    colors ... slovar, ki vsaki značilki priredi barvo, v kateri se bo izrisala

    Naredi plot kjer za vsak feature iz slovarja coeffs nariše kako se spreminja njegov koeficient pri različnih alpha vrednostih 
    """
    print(f"Number of all coeffs: {len(coeffs)}")
    # vzamemo samo featurje, ki imajo vsaj en neničelen koeficient
    # torej izločimo značilke, kjer je vrednost koeficienta pri vseh alpha vrednostih 0
    new_coeffs = dict([(elt, coeffs[elt]) for elt in coeffs if len(list(filter(lambda x: x == 0, coeffs[elt]))) != len(coeffs[elt])])
    coeffs = new_coeffs

    print(f"Number of non-zero coeffs: {len(coeffs)}")

    plt.figure()
    plt.gca().set_xscale('log')
    for i, feature in enumerate(coeffs):
        plt.plot(alphas, coeffs[feature], label=feature, color=feature_colors[feature])
    # plt.plot(alphas, np.sin(x))
    legend = plt.legend(bbox_to_anchor = (1, 1), ncol=1)
    plt.title(f'Koeificienti za {title} številka {fold_number + 1}')
    plt.xlabel('regularizacijski parameter')
    plt.ylabel('Vrednosti koeficientov')
    plt.savefig(f"lasso_plots/{title}_fold_{fold_number + 1}.png", bbox_extra_artists=(legend,), bbox_inches='tight')
    # plt.show()

    return coeffs

def make_r2_plot(r2_results, alphas, title):
    """
    r2_results ... list of lists, ki za i-ti fold vrne r2 v odvnisnosti od alpha vrednosti

    Plots r2 for each fold dashed and mean r2 solid
    """
    # compute mean r2 across all folds
    # mean_r2 = np.mean(r2_results, axis=0)

    # colors = plt.cm.tab10.colors

    # plt.figure()
    # plt.gca().set_xscale('log')

    # for i in range(len(r2_results)):
    #     plt.plot(alphas, r2_results[i], linestyle='dashed', color=colors[i])

    # plt.plot(alphas, mean_r2, color='black')
    # plt.plot(alphas, [0.9 for i in range(len(mean_r2))], color='gray', linestyle='dashed')
    # plt.title(f'R2 po prečnem preverjanju za {title}')
    # plt.xlabel('regularizacijski parameter')
    # plt.ylabel('R2')

    mean_r2 = np.mean(r2_results, axis=0)

    colors = plt.cm.tab10.colors

    plt.figure()
    ax = plt.gca()
    ax.set_xscale('log')

    for i in range(len(r2_results)):
        ax.plot(alphas, r2_results[i], linestyle='dashed', color=colors[i])
        
    ax.plot(alphas, mean_r2, color='black')
    ax.plot(alphas, [0.9 for _ in range(len(mean_r2))], color='gray', linestyle='dashed')

    # Setting custom x-axis ticks
    tick_positions = alphas[::5] + [alphas[-1]]  # Select every 5th tick + the last one
    tick_positions = [float(tick) for tick in tick_positions]
    
    # Convert tick positions to the form 10^__
    tick_labels = [f'$10^{{{round(np.log10(tick), 2)}}}$' for tick in tick_positions]
    
    # Apply the ticks and labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    # Setting the title and labels
    ax.set_title(f'R² po prečnem preverjanju za {title}')
    ax.set_xlabel('Regularizacijski parameter')
    ax.set_ylabel('R²')

    # Adding the legend
    # ax.legend(['sklop 1', 'sklop 2', 'sklop 3', 'sklop 4', 'sklop 5', 
    #            'sklop 6', 'sklop 7', 'sklop 8', 'sklop 9', 'sklop 10', 
    #            'povprečje', "0.9"], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend(['sklop 1', 'sklop 2', 'sklop 3', 'sklop 4', 'sklop 5', 'sklop 6', 'sklop 7', 'sklop 8', 'sklop 9', 'sklop 10', 'povprečje', "0.9"])
    plt.savefig(f"lasso_plots/r2_plot_{title}_11092024.png")
    plt.show()

def make_joint_plot(c_coeffs, r_coeffs, s_coeffs, alphas, c_plot, r_plot, s_plot, show_r=False, renaming=renaming, top_n=None):
    """
    Takes the coeffs computed in make_plots and plots average Vrednosti koeficientov for each feature.
    c_coeffs is a list of dictionaries where keys are feature names and values are lists of coefficient value for each alpha
    """

    # CONTRACTILITY
    # za vsak feature dobimo seznam seznamov, ki ga moramo povprečiti
    contractility = dict()
    for dictionary in c_coeffs:
        for feature in dictionary:
            if feature not in contractility:
                contractility[feature] = []
            contractility[feature].append(dictionary[feature])

    # povprečimo vrednosti
    for feature in contractility:
        contractility[feature] = np.mean(contractility[feature], axis=0)

    # remove features that have all zero coefficients
    contractility_old_names = dict((feature, contractility[feature]) for feature in contractility if not all([x == 0 for x in contractility[feature]]))
    contractility = dict()
    for feature in contractility_old_names:
        contractility[renaming[feature]] = contractility_old_names[feature]
    print(contractility)

    contractility = dict(sorted(contractility.items(), key=lambda item: max(abs(item[1])), reverse=True))

    if top_n is not None:
        contractility = dict(list(contractility.items())[:top_n])



    # RESISTANCE
    resistance = dict()
    for dictionary in r_coeffs:
        for feature in dictionary:
            if feature not in resistance:
                resistance[feature] = []
            resistance[feature].append(dictionary[feature])

    for feature in resistance:
        resistance[feature] = np.mean(resistance[feature], axis=0)

    resistance_old_names = dict((feature, resistance[feature]) for feature in resistance if not all([x == 0 for x in resistance[feature]]))

    resistance = dict()
    for feature in resistance_old_names:
        resistance[renaming[feature]] = resistance_old_names[feature]

    resistance = dict(sorted(resistance.items(), key=lambda item: max(abs(item[1])), reverse=True))

    if top_n is not None:
        resistance = dict(list(resistance.items())[:top_n])

    # STIFFNESS
    stiffness = dict()
    for dictionary in s_coeffs:
        for feature in dictionary:
            if feature not in stiffness:
                stiffness[feature] = []
            stiffness[feature].append(dictionary[feature])

    for feature in stiffness:
        stiffness[feature] = np.mean(stiffness[feature], axis=0)

    stiffness_old_names = dict((feature, stiffness[feature]) for feature in stiffness if not all([x == 0 for x in stiffness[feature]]))

    stiffness = dict()
    for feature in stiffness_old_names:
        stiffness[renaming[feature]] = stiffness_old_names[feature]

    stiffness = dict(sorted(stiffness.items(), key=lambda item: max(abs(item[1])), reverse=True))

    if top_n is not None:
        stiffness = dict(list(stiffness.items())[:top_n])

    # plotamo
    plt.figure()
    ax1 = plt.gca()
    ax1.set_xscale('log')

    # Plotting each feature in stiffness
    for feature in contractility:
        original_feature_name = [key for key, value in renaming.items() if value == feature][0]
        ax1.plot(alphas, list(contractility[feature]), label=feature, color=feature_colors[original_feature_name])

    # Setting the legend outside the plot
    legend = ax1.legend(bbox_to_anchor=(1.1, 1), ncol=1)

    # Setting custom x-axis ticks and labels
    tick_positions = alphas[::3] + [alphas[-1]]
    # print(tick_positions)
    tick_positions = [float(tick) for tick in tick_positions]

    # Convert tick positions to the form 10^__
    tick_labels = [f'$10^{{{round(np.log10(tick), 3)}}}$' for tick in tick_positions]

    # Set the ticks and labels using ax1
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)

    # Adding titles and labels
    ax1.set_title('Povprečna vrednost koeficientov za kontraktilnost')
    ax1.set_xlabel('Regularizacijski parameter')
    ax1.set_ylabel('Vrednosti koeficientov')

    # If show_r is True, add the secondary y-axis for R²
    if show_r:
        ax2 = ax1.twinx()  # Create a twin y-axis sharing the same x-axis

        mean_r2 = np.mean(c_plot, axis=0)

        max_val = np.max(mean_r2)
        alpha_80 = 0.8 * max_val
        alpha_90 = 0.9 * max_val

        # Find alpha where mean_r2 is nearest 0.8 and 0.9
        alpha_08 = alphas[np.argmin(np.abs(mean_r2 - alpha_80))]
        alpha_09 = alphas[np.argmin(np.abs(mean_r2 - alpha_90))]

        ax2.plot(alphas, mean_r2, color='black', linewidth=2, label='Povprečen R²')

        # Plot vertical lines at alpha_08 and alpha_09
        ax2.axvline(x=alpha_08, color='gray', linestyle='dashed', label='R² = 0.8')
        ax2.axvline(x=alpha_09, color='gray', linestyle='dashed', label='R² = 0.9')

        ax2.legend(loc='lower right')
        ax2.set_ylabel('R²')
        ax2.tick_params(axis='y', direction='in')
        ax2.set_ylim([0, 1])  # Fix the y-axis of ax2 between 0 and 1

        # Align zero on both y-axes
        y1_min, y1_max = ax1.get_ylim()
        y2_min, y2_max = ax2.get_ylim()

        y1_range = max(abs(y1_min), y1_max)
        y2_range = max(abs(y2_min), y2_max)

        ax1.set_ylim(-y1_range, y1_range)
        ax2.set_ylim(-y2_range, y2_range)


    plt.savefig("lasso_plots/average_contractility_FINAL.png", bbox_extra_artists=(legend,), bbox_inches='tight')

    plt.show()

    plt.figure()
    ax1 = plt.gca()
    ax1.set_xscale('log')

    # Plotting each feature in stiffness
    for feature in resistance:
        original_feature_name = [key for key, value in renaming.items() if value == feature][0]
        ax1.plot(alphas, resistance[feature], label=feature, color=feature_colors[original_feature_name])

    # Setting the legend outside the plot
    legend = ax1.legend(bbox_to_anchor=(1.1, 1), ncol=1)

    # Setting custom x-axis ticks and labels
    tick_positions = alphas[::3] + [alphas[-1]]
    # print(tick_positions)
    tick_positions = [float(tick) for tick in tick_positions]

    # Convert tick positions to the form 10^__
    tick_labels = [f'$10^{{{round(np.log10(tick), 3)}}}$' for tick in tick_positions]

    # Set the ticks and labels using ax1
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)

    # Adding titles and labels
    ax1.set_title('Povprečna vrednost koeficientov za upornost')
    ax1.set_xlabel('Regularizacijski parameter')
    ax1.set_ylabel('Vrednosti koeficientov')

    # If show_r is True, add the secondary y-axis for R²
    if show_r:
        ax2 = ax1.twinx()  # Create a twin y-axis sharing the same x-axis

        mean_r2 = np.mean(r_plot, axis=0)

        max_val = np.max(mean_r2)
        alpha_80 = 0.8 * max_val
        alpha_90 = 0.9 * max_val

        # Find alpha where mean_r2 is nearest 0.8 and 0.9
        alpha_08 = alphas[np.argmin(np.abs(mean_r2 - alpha_80))]
        alpha_09 = alphas[np.argmin(np.abs(mean_r2 - alpha_90))]

        ax2.plot(alphas, mean_r2, color='black', linewidth=2, label='Povprečen R²')

        # Plot vertical lines at alpha_08 and alpha_09
        ax2.axvline(x=alpha_08, color='gray', linestyle='dashed', label='R² = 0.8')
        ax2.axvline(x=alpha_09, color='gray', linestyle='dashed', label='R² = 0.9')

        ax2.legend(loc='lower right')
        ax2.set_ylabel('R²')
        ax2.tick_params(axis='y', direction='in')
        ax2.set_ylim([0, 1])  # Fix the y-axis of ax2 between 0 and 1

        # Align zero on both y-axes
        y1_min, y1_max = ax1.get_ylim()
        y2_min, y2_max = ax2.get_ylim()

        y1_range = max(abs(y1_min), y1_max)
        y2_range = max(abs(y2_min), y2_max)

        ax1.set_ylim(-y1_range, y1_range)
        ax2.set_ylim(-y2_range, y2_range)

    plt.savefig("lasso_plots/average_resistance_FINAL.png", bbox_extra_artists=(legend,), bbox_inches='tight')

    plt.show()

    plt.figure()
    ax1 = plt.gca()
    ax1.set_xscale('log')

    # Plotting each feature in stiffness
    for feature in stiffness:
        original_feature_name = [key for key, value in renaming.items() if value == feature][0]
        ax1.plot(alphas, stiffness[feature], label=feature, color=feature_colors[original_feature_name])

    # Setting the legend outside the plot
    legend = ax1.legend(bbox_to_anchor=(1.1, 1), ncol=1)

    # Setting custom x-axis ticks and labels
    tick_positions = alphas[::3] + [alphas[-1]]
    # print(tick_positions)
    tick_positions = [float(tick) for tick in tick_positions]

    # Convert tick positions to the form 10^__
    tick_labels = [f'$10^{{{round(np.log10(tick), 3)}}}$' for tick in tick_positions]

    # Set the ticks and labels using ax1
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)

    # Adding titles and labels
    ax1.set_title('Povprečna vrednost koeficientov za podajnost')
    ax1.set_xlabel('Regularizacijski parameter')
    ax1.set_ylabel('Vrednosti koeficientov')

    # If show_r is True, add the secondary y-axis for R²
    if show_r:
        ax2 = ax1.twinx()  # Create a twin y-axis sharing the same x-axis

        mean_r2 = np.mean(s_plot, axis=0)

        max_val = np.max(mean_r2)
        alpha_80 = 0.8 * max_val
        alpha_90 = 0.9 * max_val

        # Find alpha where mean_r2 is nearest 0.8 and 0.9
        alpha_08 = alphas[np.argmin(np.abs(mean_r2 - alpha_80))]
        alpha_09 = alphas[np.argmin(np.abs(mean_r2 - alpha_90))]

        ax2.plot(alphas, mean_r2, color='black', linewidth=2, label='Povprečen R²')

        # Plot vertical lines at alpha_08 and alpha_09
        ax2.axvline(x=alpha_08, color='gray', linestyle='dashed', label='R² = 0.8')
        ax2.axvline(x=alpha_09, color='gray', linestyle='dashed', label='R² = 0.9')

        ax2.legend(loc='lower right')
        ax2.set_ylabel('R²')
        ax2.tick_params(axis='y', direction='in')
        ax2.set_ylim([0, 1])  # Fix the y-axis of ax2 between 0 and 1

        # Align zero on both y-axes
        y1_min, y1_max = ax1.get_ylim()
        y2_min, y2_max = ax2.get_ylim()

        y1_range = max(abs(y1_min), y1_max)
        y2_range = max(abs(y2_min), y2_max)

        ax1.set_ylim(-y1_range, y1_range)
        ax2.set_ylim(-y2_range, y2_range)
    # plt.gca().set_xscale('log')
    # for feature in stiffness:
    #     plt.plot(alphas, stiffness[feature], label=feature, color=feature_colors[feature])

    # legend = plt.legend(bbox_to_anchor = (1.1, 1), ncol=1)

    # tick_positions = alphas[::5] + [alphas[-1]]
    # tick_positions = [float(tick) for tick in tick_positions]

    # # Convert tick positions to the form 10^__
    # tick_labels = [f'$10^{{{round(tick, 4)}}}$' for tick in tick_positions]

    # plt.set_xticks(tick_positions)
    # plt.set_xticklabels(tick_labels)

    # plt.title('Povprečna vrednost koeficientov za podajnost')

    # plt.xlabel('regularizacijski parameter')
    # plt.ylabel('Vrednosti koeficientov')

    # if show_r:
    #     ax1 = plt.gca()
    #     ax2 = ax1.twinx()  # Create a twin y-axis sharing the same x-axis

    #     mean_r2 = np.mean(s_plot, axis=0)

    #     max_val = np.max(mean_r2)
    #     alpha_80 = 0.8 * max_val
    #     alpha_90 = 0.9 * max_val

    #     # find alpha where mean_r2 is nearest 0.8 and the alpha where it is nearest 0.9
    #     alpha_08 = alphas[np.argmin(np.abs(mean_r2 - alpha_80))]
    #     alpha_09 = alphas[np.argmin(np.abs(mean_r2 - alpha_90))]

    #     ax2.plot(alphas, mean_r2, color='black', linewidth=2, label='povprečen R2')

    #     # plot vertical lines at alpha_08 and alpha_09
    #     ax2.axvline(x=alpha_08, color='gray', linestyle='dashed', label='R2 = 0.8')
    #     ax2.axvline(x=alpha_09, color='gray', linestyle='dashed', label='R2 = 0.9')

    #     ax2.legend(loc='upper right')
    #     ax2.set_ylabel('R2')
    #     ax2.tick_params(axis='y', direction='in')
    #     # ax2.set_ylim([0, 1])

    #     y1_min, y1_max = ax1.get_ylim()
    #     y2_min, y2_max = ax2.get_ylim()

    #     # Make sure zero is at the same height
    #     # Calculate new limits for both axes
    #     y1_range = max(abs(y1_min), y1_max)
    #     y2_range = max(abs(y2_min), y2_max)

    #     # Set the limits to ensure zero aligns
    #     ax1.set_ylim(-y1_range, y1_range)
    #     ax2.set_ylim(-y2_range, y2_range)

    plt.savefig("lasso_plots/average_stiffness_FINAL.png", bbox_extra_artists=(legend,), bbox_inches='tight')

    plt.show()



def make_plots(splits, alphas, features=feature_names):
    """
    imamo splits, ki je seznam 10 splitov za cross validation
    za vsak split naredim single plot koeficientov
    in na koncu še skupni r2 plot
    """
    c_plot, r_plot, s_plot = [], [], []
    c_coeffs, r_coeffs, s_coeffs = [], [], []
    lasso_report_c, lasso_report_r, lasso_report_s = [], [], []

    for i, split in enumerate(splits):
        X_train, X_test, C_train, C_test, R_train, R_test, S_train, S_test = split

        results_C = lasso_regression(X_train, C_train, X_test, C_test, alphas)
        results_R = lasso_regression(X_train, R_train, X_test, R_test, alphas)
        results_S = lasso_regression(X_train, S_train, X_test, S_test, alphas)

        alphas, c_coef = parse_coeffs(results_C, features)
        alphas, r_coef = parse_coeffs(results_R, features)
        alphas, s_coef = parse_coeffs(results_S, features)

        c_coeffs.append(c_coef)
        r_coeffs.append(r_coef)
        s_coeffs.append(s_coef)

        # naredimo plot za koeficiente
        # print(c_coef)
        # coeffs_c = make_single_plot(c_coef, alphas, title="CONTRACTILITY", fold_number=i)
        # coeffs_r = make_single_plot(r_coef, alphas, title="RESISTANCE", fold_number=i)
        # coeffs_s = make_single_plot(s_coef, alphas, title="STIFFNESS", fold_number=i)

        # tole bomo risali črtkano
        c_r2 = [results_C[alpha]['r2'] for alpha in results_C]
        r_r2 = [results_R[alpha]['r2'] for alpha in results_R]
        s_r2 = [results_S[alpha]['r2'] for alpha in results_S]

        c_plot.append(c_r2)
        r_plot.append(r_r2)
        s_plot.append(s_r2)

        # lasso_report_c.append("=====================================================")
        # lasso_report_c.append(f"Non zero features for contractility for fold number {i+1} are:")
        # lasso_report_c += [feature for feature in coeffs_c]

        # lasso_report_r.append("=====================================================")
        # lasso_report_r.append(f"Non zero features for resistance for fold number {i+1} are:")
        # lasso_report_r += [feature for feature in coeffs_r]

        # lasso_report_s.append("=====================================================")
        # lasso_report_s.append(f"Non zero features for stiffness for fold number {i+1} are:")
        # lasso_report_s += [feature for feature in coeffs_s]

    print(c_coeffs)
    make_joint_plot(c_coeffs, r_coeffs, s_coeffs, alphas, c_plot, r_plot, s_plot, show_r=True)

    # print(lasso_report)
    filename_c = "lasso_plots/lasso_report_c.txt"
    filename_r = "lasso_plots/lasso_report_r.txt"
    filename_s = "lasso_plots/lasso_report_s.txt"
    with open(filename_c, 'w') as file:
        for row in lasso_report_c:
            file.write(row)
            file.write("\n")
    
    with open(filename_r, 'w') as file:
        for row in lasso_report_r:
            file.write(row)
            file.write("\n")

    with open(filename_s, 'w') as file:
        for row in lasso_report_s:
            file.write(row)
            file.write("\n")

    # naredimo plot za r2
    make_r2_plot(c_plot, alphas, title="KONTRAKTILNOST")
    make_r2_plot(r_plot, alphas, title="UPORNOST")
    make_r2_plot(s_plot, alphas, title="PODAJNOST")


def compute_results(splits, alphas, features=feature_names):

    all_results_C, all_results_R, all_results_S = [], [], []

    for i, split in enumerate(splits):
        X_train, X_test, C_train, C_test, R_train, R_test, S_train, S_test = split

        results_C = lasso_regression(X_train, C_train, X_test, C_test, alphas)
        results_R = lasso_regression(X_train, R_train, X_test, R_test, alphas)
        results_S = lasso_regression(X_train, S_train, X_test, S_test, alphas)

        all_results_C.append(results_C)
        all_results_R.append(results_R)
        all_results_S.append(results_S)

    with open("data/lasso_results/lasso_results_C_FINAL.json", "w") as file:
        json.dump(all_results_C, file)

    with open("data/lasso_results/lasso_results_R_FINAL.json", "w") as file:
        json.dump(all_results_R, file)

    with open("data/lasso_results/lasso_results_S_FINAL.json", "w") as file:
        json.dump(all_results_S, file)

    return all_results_C, all_results_R, all_results_S

def compute_results_for_S(splits, alphas, features=feature_names):

    all_results_C, all_results_R, all_results_S = [], [], []

    for i, split in enumerate(splits):
        X_train, X_test, C_train, C_test, R_train, R_test, S_train, S_test = split

        # results_C = lasso_regression(X_train, C_train, X_test, C_test, alphas)
        # results_R = lasso_regression(X_train, R_train, X_test, R_test, alphas)
        results_S = lasso_regression(X_train, S_train, X_test, S_test, alphas)

        # all_results_C.append(results_C)
        # all_results_R.append(results_R)
        all_results_S.append(results_S)

    # with open("data/lasso_results/lasso_results_C_11092024.json", "w") as file:
    #     json.dump(all_results_C, file)

    # with open("data/lasso_results/lasso_results_R_11092024.json", "w") as file:
    #     json.dump(all_results_R, file)

    with open("data/lasso_results/lasso_results_S_11092024_new.json", "w") as file:
        json.dump(all_results_S, file)

    return all_results_S

def load_results():

    with open("data/lasso_results/lasso_results_C_FINAL.json", "r") as file:
        all_results_C = json.load(file)

    with open("data/lasso_results/lasso_results_R_FINAL.json", "r") as file:
        all_results_R = json.load(file)

    with open("data/lasso_results/lasso_results_S_FINAL.json", "r") as file:
        all_results_S = json.load(file)

    return all_results_C, all_results_R, all_results_S


def make_plots_from_results(alphas, all_results_C, all_results_R, all_results_S, features=feature_names, k= 10, top_n=None):
    """
    imamo splits, ki je seznam 10 splitov za cross validation
    za vsak split naredim single plot koeficientov
    in na koncu še skupni r2 plot
    """
    c_plot, r_plot, s_plot = [], [], []
    c_coeffs, r_coeffs, s_coeffs = [], [], []
    lasso_report_c, lasso_report_r, lasso_report_s = [], [], []

    for i in range(k):
        # X_train, X_test, C_train, C_test, R_train, R_test, S_train, S_test = split
        
        results_C = all_results_C[i]
        results_R = all_results_R[i]
        results_S = all_results_S[i]

        alphas, c_coef = parse_coeffs(results_C, features)
        alphas, r_coef = parse_coeffs(results_R, features)
        alphas, s_coef = parse_coeffs(results_S, features)

        c_coeffs.append(c_coef)
        r_coeffs.append(r_coef)
        s_coeffs.append(s_coef)

        # naredimo plot za koeficiente
        # print(c_coef)
        # coeffs_c = make_single_plot(c_coef, alphas, title="KONTRAKTILNOST", fold_number=i)
        # coeffs_r = make_single_plot(r_coef, alphas, title="UPORNOST", fold_number=i)
        # coeffs_s = make_single_plot(s_coef, alphas, title="PODAJNOST", fold_number=i)

        # tole bomo risali črtkano
        c_r2 = [results_C[alpha]['r2'] for alpha in results_C]
        r_r2 = [results_R[alpha]['r2'] for alpha in results_R]
        s_r2 = [results_S[alpha]['r2'] for alpha in results_S]

        c_plot.append(c_r2)
        r_plot.append(r_r2)
        s_plot.append(s_r2)

        # lasso_report_c.append("=====================================================")
        # lasso_report_c.append(f"Non zero features for contractility for fold number {i+1} are:")
        # lasso_report_c += [feature for feature in coeffs_c]

        # lasso_report_r.append("=====================================================")
        # lasso_report_r.append(f"Non zero features for resistance for fold number {i+1} are:")
        # lasso_report_r += [feature for feature in coeffs_r]

        # lasso_report_s.append("=====================================================")
        # lasso_report_s.append(f"Non zero features for stiffness for fold number {i+1} are:")
        # lasso_report_s += [feature for feature in coeffs_s]

    make_joint_plot(c_coeffs, r_coeffs, s_coeffs, alphas, c_plot, r_plot, s_plot, show_r=True, top_n=top_n)

    # print(lasso_report)
    filename_c = "lasso_plots/lasso_report_c.txt"
    filename_r = "lasso_plots/lasso_report_r.txt"
    filename_s = "lasso_plots/lasso_report_s.txt"
    with open(filename_c, 'w') as file:
        for row in lasso_report_c:
            file.write(row)
            file.write("\n")
    
    with open(filename_r, 'w') as file:
        for row in lasso_report_r:
            file.write(row)
            file.write("\n")

    with open(filename_s, 'w') as file:
        for row in lasso_report_s:
            file.write(row)
            file.write("\n")

    # naredimo plot za r2
    make_r2_plot(c_plot, alphas, title="KONTRAKTILNOST")
    make_r2_plot(r_plot, alphas, title="UPORNOST")
    make_r2_plot(s_plot, alphas, title="PODAJNOST")


if __name__=="__main__":

    X, C, R, S, feature_names = load_new_data()

    # orig
    alphas = list(np.logspace(-3, 0.6, 50))
    alphas = np.logspace(-3, 0.6, 50).tolist()

    # splits = split_dataset(X, C, R, S, mode='cv', k=10)

    # all_results_C, all_results_R, all_results_S = compute_results(splits, alphas, feature_names)
    # all_results_S = compute_results_for_S(splits, alphas, feature_names)

    all_results_C, all_results_R, all_results_S = load_results()



    make_plots_from_results(alphas, all_results_C, all_results_R, all_results_S, feature_names, top_n=None)

    # make_plots(splits, alphas)

