import numpy as np
import pandas as pd
import json

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, make_scorer, precision_score, log_loss

# from feature_definition.feature_definition_framework import FEATURES

from feature_renaming import renaming

import matplotlib.pyplot as plt

# from lasso_regression import lasso_regression, parse_coeffs, split_dataset


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



def standardize(X_train, X_test):
    """By standardizing the input data, we ensure that each feature contributes equally to the model's predictions"""

    # združi X_train in X_test
    X_mean = np.mean(X_train, axis=0) # mean of columns
    X_std = np.std(X_train, axis=0)
    X_std[X_std == 0] = 1 # to avoid division by zero replace zero values with 1
    X_std = X_std.reshape(1, -1) # instead of a list make it a column

    # use the same values on test data to use the same scaling as we did on train data!
    X_train, X_test = (X_train - X_mean) / X_std, (X_test - X_mean) / X_std

    return X_train, X_test


def split_dataset(X, C, R, S, mode=None, size=None, k=None, seed=42):
    """
    train / test split ... naključen train test split
    cv ... razdeli na k delob

    k ... število delov oziroma size of test set
    """
    n = len(X)
    one_fold = n // k
    splits = []

    column_names = X.columns.tolist()
    X = np.array(X.values.tolist()) 
    C = np.array(C.values.tolist()) 
    R = np.array(R.values.tolist()) 
    S = np.array(S.values.tolist()) 

    split_data = np.hstack([np.ones(one_fold)*i for i in range(1, k+1)])
    np.random.seed(seed)
    np.random.shuffle(split_data)

    C = np.array(C)
    R = np.array(R)
    S = np.array(S)

    if mode == 'train_test':

        j = np.random.randint(1, k+1) # j-th fold

        train_indices = [i for i in range(len(split_data)) if split_data[i] != j] # get indices that are NOT in the j-th fold
        test_indices = [i for i in range(len(split_data)) if split_data[i] == j] # get indices that ARE in the j-th fold

        X_train, X_test = X[train_indices], X[test_indices] # split X into 2 parts

        # standardize
        X_train, X_test = standardize(X_train, X_test)

        C_train, C_test = C[train_indices], C[test_indices] # split y into 2 parts
        R_train, R_test = R[train_indices], R[test_indices] # split y into 2 parts
        S_train, S_test = S[train_indices], S[test_indices] # split y into 2 parts

        X_train = pd.DataFrame(X_train, columns=column_names)
        X_test = pd.DataFrame(X_test, columns=column_names)

        C_train = pd.Series(C_train, name='C')
        C_test = pd.Series(C_test, name='C')

        R_train = pd.Series(R_train, name='R')
        R_test = pd.Series(R_test, name='R')

        S_train = pd.Series(S_train, name='S')
        S_test = pd.Series(S_test, name='S')

        splits.append((X_train, X_test, C_train, C_test, R_train, R_test, S_train, S_test))

    
    if mode == 'cv':

        for j in range(1, k+1): # j-th fold

            train_indices = [i for i in range(len(split_data)) if split_data[i] != j] # get indices that are NOT in the j-th fold
            test_indices = [i for i in range(len(split_data)) if split_data[i] == j] # get indices that ARE in the j-th fold

            X_train, X_test = X[train_indices], X[test_indices] # split X into 2 parts

            # standardize
            X_train, X_test = standardize(X_train, X_test)

            C_train, C_test = C[train_indices], C[test_indices] # split y into 2 parts
            R_train, R_test = R[train_indices], R[test_indices] # split y into 2 parts
            S_train, S_test = S[train_indices], S[test_indices] # split y into 2 parts

            X_train = pd.DataFrame(X_train, columns=column_names)
            X_test = pd.DataFrame(X_test, columns=column_names)

            C_train = pd.Series(C_train, name='C')
            C_test = pd.Series(C_test, name='C')

            R_train = pd.Series(R_train, name='R')
            R_test = pd.Series(R_test, name='R')

            S_train = pd.Series(S_train, name='S')
            S_test = pd.Series(S_test, name='S')

            splits.append((X_train, X_test, C_train, C_test, R_train, R_test, S_train, S_test))

    return splits


def lasso_regression(X_train, y_train, X_test, y_test, alphas):
    """
    Performs lasso regression for X and y.
    If we are using the whole set then we set X_test and y_test to None.

    Returns for each alpha: coefficients, r2, mse, accuracy, precision, log_loss
    """

    results = dict()

    for alpha in alphas:

        model = Lasso(alpha=alpha, max_iter=1000000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_r2 = r2_score(y_test, y_pred)
        y_mse = mean_squared_error(y_test, y_pred)
        # C_accuracy = accuracy_score(C, C_pred)
        y_accuracy = 0
        y_log_loss = 0 # log_loss(y_test, y_pred)
        print(f"Results: R2={y_r2}, MSE={y_mse}, Accuracy={y_accuracy}")
        coeffs = model.coef_

        results[alpha] = {
            "coeffs": list(coeffs),
            "r2": y_r2,
            "mse": y_mse,
            "accuracy": y_accuracy,
            "log_loss": y_log_loss
        }

    return results

def parse_coeffs(results, feature_names):
    
    """
    Parses the coefficients from the results.

    Returns list of alphas (x axis) and dictionary of coefficients for each feature (y axis).
    """
    alphas = list(results.keys())
    coeffs = dict((feature_names[i], [results[alpha]['coeffs'][i] for alpha in results]) for i in range(len(feature_names)))

    return alphas, coeffs


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

def load_results():

    with open("data/lasso_results/lasso_results_C_FINAL.json", "r") as file:
        all_results_C = json.load(file)

    with open("data/lasso_results/lasso_results_R_FINAL.json", "r") as file:
        all_results_R = json.load(file)

    with open("data/lasso_results/lasso_results_S_FINAL.json", "r") as file:
        all_results_S = json.load(file)

    return all_results_C, all_results_R, all_results_S



def make_r2_plot(r2_results, alphas, title):
    """
    r2_results ... list of lists, ki za i-ti fold vrne r2 v odvnisnosti od alpha vrednosti

    Plots r2 for each fold dashed and mean r2 solid
    """


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
    plt.savefig(f"lasso_plots/r2_plot_{title}_FINAL.png")
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

    plt.savefig("lasso_plots/average_stiffness_FINAL.png", bbox_extra_artists=(legend,), bbox_inches='tight')

    plt.show()


def make_joint_plot2(c_coeffs, r_coeffs, s_coeffs, alphas, c_plot, r_plot, s_plot, show_r=False, renaming=renaming, top_n=None):
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
    # ax1.set_xscale('log')

    # Setting custom x-axis ticks and labels
    third_tick_positions = alphas[::4] #+ [alphas[-1]]
    tick_positions = alphas
    print(len(tick_positions))
    # print(tick_positions)
    tick_positions = [float(tick) for tick in tick_positions]
    third_tick_positions = [float(tick) for tick in third_tick_positions]
    print(len(tick_positions))

    # Convert tick positions to the form 10^__
    tick_labels = [f'$10^{{{round(np.log10(tick), 2)}}}$' if tick in third_tick_positions else '' for tick in tick_positions]
    print(len(tick_labels))

    # # Set the ticks and labels using ax1
    # ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)

    # Plotting each feature in stiffness
    for feature in contractility:
        original_feature_name = [key for key, value in renaming.items() if value == feature][0]
        ax1.plot(alphas, list(contractility[feature]), label=feature, color=feature_colors[original_feature_name])
        # print(len(contractility[feature]))
    # Setting the legend outside the plot
    # legend = ax1.legend(bbox_to_anchor=(1.1, 1), ncol=1)

    

    # Adding titles and labels
    ax1.set_title('Povprečna vrednost koeficientov za kontraktilnost')
    ax1.set_xlabel('Regularizacijski parameter')
    ax1.set_ylabel('Vrednosti koeficientov')

    # If show_r is True, add the secondary y-axis for R²
    if show_r:
        ax2 = ax1.twinx()  # Create a twin y-axis sharing the same x-axis
        # ax2.set_xscale('log')

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

        # ax2.legend(loc='lower right')
        ax2.set_ylabel('R²')
        ax2.tick_params(axis='y', direction='in')
        ax2.set_ylim([0, 1])  # Fix the y-axis of ax2 between 0 and 1

        # tick_positions = alphas[::3] #+ [alphas[-1]]
        # # tick_positions = alphas
        # print(len(tick_positions))
        # # print(tick_positions)
        # tick_positions = [float(tick) for tick in tick_positions]
        # print(len(tick_positions))

        # # Convert tick positions to the form 10^__
        # tick_labels = [f'$10^{{{round(np.log10(tick), 3)}}}$' for tick in tick_positions]
        # print(len(tick_labels))

        # # Set the ticks and labels using ax1
        # ax2.set_xticks(tick_positions)
        # ax2.set_xticklabels(tick_labels)

        # Align zero on both y-axes
        y1_min, y1_max = ax1.get_ylim()
        y2_min, y2_max = ax2.get_ylim()

        y1_range = max(abs(y1_min), y1_max)
        y2_range = max(abs(y2_min), y2_max)

        ax1.set_ylim(-y1_range, y1_range)
        ax2.set_ylim(-y2_range, y2_range)


    plt.savefig("lasso_plots/average_contractility_FINAL_wo_legend.png")#, bbox_extra_artists=(legend,), bbox_inches='tight')

    plt.show()

    plt.figure()
    ax1 = plt.gca()
    # ax1.set_xscale('log')

    # Setting custom x-axis ticks and labels
    third_tick_positions = alphas[::4] #+ [alphas[-1]]
    tick_positions = alphas
    print(len(tick_positions))
    # print(tick_positions)
    tick_positions = [float(tick) for tick in tick_positions]
    third_tick_positions = [float(tick) for tick in third_tick_positions]
    print(len(tick_positions))

    # Convert tick positions to the form 10^__
    tick_labels = [f'$10^{{{round(np.log10(tick), 2)}}}$' if tick in third_tick_positions else '' for tick in tick_positions]
    print(len(tick_labels))

    # # Set the ticks and labels using ax1
    # ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)

    # Plotting each feature in stiffness
    for feature in resistance:
        original_feature_name = [key for key, value in renaming.items() if value == feature][0]
        ax1.plot(alphas, resistance[feature], label=feature, color=feature_colors[original_feature_name])

    # Setting the legend outside the plot
    # legend = ax1.legend(bbox_to_anchor=(1.1, 1), ncol=1)

    # Setting custom x-axis ticks and labels
    # tick_positions = alphas[::3] + [alphas[-1]]
    # # print(tick_positions)
    # tick_positions = [float(tick) for tick in tick_positions]

    # # Convert tick positions to the form 10^__
    # tick_labels = [f'$10^{{{round(np.log10(tick), 3)}}}$' for tick in tick_positions]

    # # Set the ticks and labels using ax1
    # ax1.set_xticks(tick_positions)
    # ax1.set_xticklabels(tick_labels)

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

        # ax2.legend(loc='lower right')
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

    plt.savefig("lasso_plots/average_resistance_FINAL_wo_legend.png")#, bbox_extra_artists=(legend,), bbox_inches='tight')

    plt.show()

    plt.figure()
    ax1 = plt.gca()
    # ax1.set_xscale('log')

    
    # Setting custom x-axis ticks and labels
    third_tick_positions = alphas[::4] #+ [alphas[-1]]
    tick_positions = alphas
    print(len(tick_positions))
    # print(tick_positions)
    tick_positions = [float(tick) for tick in tick_positions]
    third_tick_positions = [float(tick) for tick in third_tick_positions]
    print(len(tick_positions))

    # Convert tick positions to the form 10^__
    tick_labels = [f'$10^{{{round(np.log10(tick), 2)}}}$' if tick in third_tick_positions else '' for tick in tick_positions]
    print(len(tick_labels))

    # # Set the ticks and labels using ax1
    # ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)

    # Plotting each feature in stiffness
    for feature in stiffness:
        original_feature_name = [key for key, value in renaming.items() if value == feature][0]
        ax1.plot(alphas, stiffness[feature], label=feature, color=feature_colors[original_feature_name])

    # Setting the legend outside the plot
    # legend = ax1.legend(bbox_to_anchor=(1.1, 1), ncol=1)

    # # Setting custom x-axis ticks and labels
    # tick_positions = alphas[::3] + [alphas[-1]]
    # # print(tick_positions)
    # tick_positions = [float(tick) for tick in tick_positions]

    # # Convert tick positions to the form 10^__
    # tick_labels = [f'$10^{{{round(np.log10(tick), 3)}}}$' for tick in tick_positions]

    # Set the ticks and labels using ax1
    # ax1.set_xticks(tick_positions)
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

        # ax2.legend(loc='lower right')
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

    plt.savefig("lasso_plots/average_stiffness_FINAL_wo_legend.png")#, bbox_extra_artists=(legend,), bbox_inches='tight')

    plt.show()

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

        # tole bomo risali črtkano
        c_r2 = [results_C[alpha]['r2'] for alpha in results_C]
        r_r2 = [results_R[alpha]['r2'] for alpha in results_R]
        s_r2 = [results_S[alpha]['r2'] for alpha in results_S]

        c_plot.append(c_r2)
        r_plot.append(r_r2)
        s_plot.append(s_r2)


    make_joint_plot2(c_coeffs, r_coeffs, s_coeffs, alphas, c_plot, r_plot, s_plot, show_r=True, top_n=top_n)

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
    # make_r2_plot(c_plot, alphas, title="KONTRAKTILNOST")
    # make_r2_plot(r_plot, alphas, title="UPORNOST")
    # make_r2_plot(s_plot, alphas, title="PODAJNOST")




# # create new color dict with the same keys as above using tab20 colorscale + yellow and salmon if you run out of colors
# feature_colors = {



if __name__=="__main__":

    # X, C, R, S, feature_names = load_new_data()

    # orig
    alphas = list(np.logspace(-3, 0.6, 50))
    alphas = np.logspace(-3, 0.6, 50).tolist()

    # splits = split_dataset(X, C, R, S, mode='cv', k=10)

    # all_results_C, all_results_R, all_results_S = compute_results(splits, alphas, feature_names)


    all_results_C, all_results_R, all_results_S = load_results()

    make_plots_from_results(alphas, all_results_C, all_results_R, all_results_S,  feature_names, top_n=None)

    # make_plots(splits, alphas)