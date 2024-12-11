import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from lasso_regression import split_dataset

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


def perform_linear_regression(X_train, y_train, X_test, y_test):

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict y using the trained model
    y_pred = model.predict(X_test)

    # Compute R-squared
    r2 = r2_score(y_test, y_pred)

    return r2

def linear_regression_cv(splits):

    C_r2s, R_r2s, S_r2s = [], [], []

    for i, split in enumerate(splits):
        X_train, X_test, C_train, C_test, R_train, R_test, S_train, S_test = split

        r2_C = perform_linear_regression(X_train, C_train, X_test, C_test)
        r2_R = perform_linear_regression(X_train, R_train, X_test, R_test)
        r2_S = perform_linear_regression(X_train, S_train, X_test, S_test)

        C_r2s.append(r2_C)
        R_r2s.append(r2_R)
        S_r2s.append(r2_S)

    return C_r2s, R_r2s, S_r2s




if __name__=="__main__":

    X, C, R, S, feature_names = load_new_data()

    splits = split_dataset(X, C, R, S, mode='cv', k=10)

    C_r2s, R_r2s, S_r2s = linear_regression_cv(splits)

    print("============ ALL FEATURES ============")

    print(f"Average R² for C: {np.mean(C_r2s):.4f}")
    print(f"Average R² for R: {np.mean(R_r2s):.4f}")
    print(f"Average R² for S: {np.mean(S_r2s):.4f}")

    less_features = [
        # 'PP',
        'SBP',
        'DBP',
        # 'MAP',
        # # 'bpdn',
        # # 'čas do SP',
        # 'maksimalni naklon'
    ]

    X_less = X[less_features]

    splits = split_dataset(X_less, C, R, S, mode='cv', k=10)

    C_r2s, R_r2s, S_r2s = linear_regression_cv(splits)

    print("============ LESS FEATURES (sbp, dbp) ============")

    print(f"Average R² for C with less features: {np.mean(C_r2s):.4f}")
    print(f"Average R² for R with less features: {np.mean(R_r2s):.4f}")
    print(f"Average R² for S with less features: {np.mean(S_r2s):.4f}")

    less_features = [
        'PP',
        # 'SBP',
        # 'DBP',
        'MAP',
        # # 'bpdn',
        # # 'čas do SP',
        # 'maksimalni naklon'
    ]

    X_less = X[less_features]

    splits = split_dataset(X_less, C, R, S, mode='cv', k=10)

    C_r2s, R_r2s, S_r2s = linear_regression_cv(splits)

    print("============ LESS FEATURES (pp, map) ============")

    print(f"Average R² for C with less features: {np.mean(C_r2s):.4f}")
    print(f"Average R² for R with less features: {np.mean(R_r2s):.4f}")
    print(f"Average R² for S with less features: {np.mean(S_r2s):.4f}")

    less_features = [
        'PP',
        'SBP',
        'DBP',
        'MAP',
        # # 'bpdn',
        # # 'čas do SP',
        # 'maksimalni naklon'
    ]

    X_less = X[less_features]

    splits = split_dataset(X_less, C, R, S, mode='cv', k=10)

    C_r2s, R_r2s, S_r2s = linear_regression_cv(splits)

    print("============ LESS FEATURES (pp, map, dbp, sbp) ============")

    print(f"Average R² for C with less features: {np.mean(C_r2s):.4f}")
    print(f"Average R² for R with less features: {np.mean(R_r2s):.4f}")
    print(f"Average R² for S with less features: {np.mean(S_r2s):.4f}")

    less_features = [
        'PP',
        'SBP',
        'DBP',
        'MAP',
        # # 'bpdn',
        'čas do SP',
        # 'maksimalni naklon'
    ]

    X_less = X[less_features]

    splits = split_dataset(X_less, C, R, S, mode='cv', k=10)

    C_r2s, R_r2s, S_r2s = linear_regression_cv(splits)

    print("============ LESS FEATURES (pp, map, dbp, sbp, t_Sp) ============")

    print(f"Average R² for C with less features: {np.mean(C_r2s):.4f}")
    print(f"Average R² for R with less features: {np.mean(R_r2s):.4f}")
    print(f"Average R² for S with less features: {np.mean(S_r2s):.4f}")

    