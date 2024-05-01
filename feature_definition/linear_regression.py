import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score



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
    
    # add C_pred and R_pred as columns to X
    # X['C_pred'] = C_pred
    # X['R_pred'] = R_pred

    model = LinearRegression()
    model.fit(X, S)
    S_pred = model.predict(X)
    S_r2 = r2_score(S, S_pred)
    S_mse = mean_squared_error(S, S_pred)
    # S_accuracy = accuracy_score(S, S_pred)
    S_accuracy = 0
    print(f"Compliance: R2={S_r2}, MSE={S_mse}, Accuracy={S_accuracy}")
    

def linear_regression_random_split(X, C, R, S, test_size=0.2, random_state=42):
    """
    """
    X_train, X_test, C_train, C_test, R_train, R_test, S_train, S_test = train_test_split(X, C, R, S, test_size=test_size)

 

def linear_regression_cross_validation(X, C, R, S, k=[2, 3, 5, 10, 100]):
    """
    """
    
if __name__=="__main__":

    FEATURES = [
        "MAP",
        "PP",
        "SBP",
        "DBP",
        "rDNP",
        "DP",
        # "t_sys",
        "total_area",
        "systolic_area",
        "diastolic_area",
        "relative_total_area",
        "relative_systolic_area",
        "relative_diastolic_area",
        "stroke_volume",
        "cardiac_output",
        "duration_systole",
        "duration_diastole",
        "cven",
        "systolic_upstroke_slope",
        "systolic_downstroke_slope",
        "diff_sbp",
        "diff_dbp"
        # "residuum"
    ]
    X, C, R, S = load_data(FEATURES)

    linear_regression(X, C, R, S)
    # linear_regression_random_split(X, C, R, S)
    # linear_regression_cross_validation(X, C, R, S)