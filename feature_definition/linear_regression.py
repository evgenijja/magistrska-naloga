import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, make_scorer, precision_score, log_loss

from magistrska_koda.feature_definition.feature_definition_framework_old import FEATURES



def load_data(feature_names, file_path="data/features.xlsx"):
    """
    Load data from excel file.

    feature_names povem posebej zato, ker me bodo mogoče zanimali samo določeni featureji.
    """
    df = pd.read_excel(file_path)
    df.dropna(inplace=True)
    C, R, S = df['C'], df['R'], df['S']
    # X = df.drop(['C', 'R', 'S'], axis=1)
    X = df[feature_names]

    return X, C, R, S

def linear_regression(X, C, R, S):
    """
    Vzamemo vse podatke in na njih izvedemo linearno regresijo za vsak parameter posebej
    """
    model = LinearRegression()
    model.fit(X, C)
    C_pred = model.predict(X)
    C_r2 = r2_score(C, C_pred)
    C_mse = mean_squared_error(C, C_pred)
    # C_accuracy = accuracy_score(C, C_pred)
    C_accuracy = 0
    print(f"Contractility: R2={C_r2}, MSE={C_mse}, Accuracy={C_accuracy}")

    model = LinearRegression()
    model.fit(X, R)
    R_pred = model.predict(X)
    R_r2 = r2_score(R, R_pred)
    R_mse = mean_squared_error(R, R_pred)
    # R_accuracy = accuracy_score(R, R_pred)
    R_accuracy = 0
    print(f"Resistance: R2={R_r2}, MSE={R_mse}, Accuracy={R_accuracy}")

    model = LinearRegression()
    model.fit(X, S)
    S_pred = model.predict(X)
    S_r2 = r2_score(S, S_pred)
    S_mse = mean_squared_error(S, S_pred)
    # S_accuracy = accuracy_score(S, S_pred)
    S_accuracy = 0
    print(f"Compliance: R2={S_r2}, MSE={S_mse}, Accuracy={S_accuracy}")
    
# scoring_functions = {'r2': make_scorer(r2_score), 
#                      'accuracy': make_scorer(accuracy_score),
#                      'precision': make_scorer(precision_score),
#                      'log_loss': make_scorer(log_loss)}

def linear_regression_random_split(X, C, R, S, test_size=0.2, random_state=42, k_list=[2, 5, 10, 100]):
    """
    """
    X_train, X_test, C_train, C_test, R_train, R_test, S_train, S_test = train_test_split(X, C, R, S, test_size=test_size)

    print("===================================================")
    print("=================== CONTRACTILITY ===================")
    print("===================================================")
    model = LinearRegression()
    model.fit(X_train, C_train)
    C_pred = model.predict(X_test)
    C_r2 = r2_score(C_test, C_pred)
    C_mse = mean_squared_error(C_test, C_pred)
    # C_accuracy = accuracy_score(C, C_pred)
    C_accuracy = 0
    print("Na vseh podatkih:")
    print(f"Contractility: R2={C_r2}, MSE={C_mse}, Accuracy={C_accuracy}")
    for k in k_list:
        print("===================================================")
        print(f"Cross validation za k = {k}:")
        
        r2_scores = cross_val_score(model, X, C, cv=k, scoring='r2')
        print("Mean R^2:", r2_scores.mean())
        print("Standard Deviation of R^2:", r2_scores.std())

        mse_scores = cross_val_score(model, X, C, cv=k, scoring='neg_mean_squared_error')
        print("Mean MSE:", -mse_scores.mean())
        print("Standard Deviation of MSE:", mse_scores.std())

    print("=================== RESISTANCE ===================")

    model = LinearRegression()
    model.fit(X_train, R_train)
    R_pred = model.predict(X_test)
    R_r2 = r2_score(R_test, R_pred)
    R_mse = mean_squared_error(R_test, R_pred)
    # R_accuracy = accuracy_score(R, R_pred)
    R_accuracy = 0
    print(f"Resistance: R2={R_r2}, MSE={R_mse}, Accuracy={R_accuracy}")

    for k in k_list:
        print("===================================================")
        print(f"Cross validation za k = {k}:")
        
        r2_scores = cross_val_score(model, X, R, cv=k, scoring='r2')
        print("Mean R^2:", r2_scores.mean())
        print("Standard Deviation of R^2:", r2_scores.std())

        mse_scores = cross_val_score(model, X, R, cv=k, scoring='neg_mean_squared_error')
        print("Mean MSE:", -mse_scores.mean())
        print("Standard Deviation of MSE:", mse_scores.std())

    print("=================== COMPLIANCE ===================")

    model = LinearRegression()
    model.fit(X_train, S_train)
    S_pred = model.predict(X_test)
    S_r2 = r2_score(S_test, S_pred)
    S_mse = mean_squared_error(S_test, S_pred)
    # S_accuracy = accuracy_score(S, S_pred)
    S_accuracy = 0
    print(f"Compliance: R2={S_r2}, MSE={S_mse}, Accuracy={S_accuracy}")

    for k in k_list:
        print("===================================================")
        print(f"Cross validation za k = {k}:")
        
        r2_scores = cross_val_score(model, X, S, cv=k, scoring='r2')
        print("Mean R^2:", r2_scores.mean())
        print("Standard Deviation of R^2:", r2_scores.std())

        mse_scores = cross_val_score(model, X, S, cv=k, scoring='neg_mean_squared_error')
        print("Mean MSE:", -mse_scores.mean())
        print("Standard Deviation of MSE:", mse_scores.std())

# def linear_regression_cross_validation(X, C, R, S, k=[2, 3, 5, 10, 100]):
#     """
#     """

    
if __name__=="__main__":

#     FEATURES = [
#     "MAP",
#     "PP",
#     "SBP",
#     "DBP",
#     "rDNP",
#     "mean",
#     "t_sys",
#     "dicrotic_notch_idx",
#     "razmerje1",
#     "razmerje2",
#     "total area",
#     "systolic area",
#     "peak to dicrotic notch area",
#     "dicrotic notch to end area",
#     "max naklon",
#     "min naklon",
#     "max 2nd deriv after peak"


# ]
    
    X, C, R, S = load_data(FEATURES)

    # linear_regression(X, C, R, S)
    linear_regression_random_split(X, C, R, S)
    # linear_regression_cross_validation(X, C, R, S)