import numpy as np
import pandas as pd

from reading_data import *


"""
Imamo več možnih valov, na katerih lahko definiramo featurje. 

Lahko jih definiramo na:
- navadnih valovih
- valovih, ki so bili translirani
- valovih, ki so bili translirani in normalizirani
- ostanku, ki ga dobimo, če od navadnih transliranih valov odštejemo stabilen val
- ostanku, ki ga dobimo, če od transliranih in normaliziranih valov odštejemo stabilen val


"""
# vrne seznam slovarjev, kjer je vsak slovar en val skupaj s parametri
data = extract_parts_for_feature_definitions()

FEATURES = [
    "MAP",
    "PP",
    "SBP",
    # "DBP"
]

def compute_feature(part, feature_name):
    """
    Za posamezen part zračunamo nek feature.
    """
    if feature_name == "MAP":
        return np.mean(part)
    elif feature_name == "PP":
        return np.max(part) - np.min(part)
    elif feature_name == "SBP":
        return np.max(part)
    elif feature_name == "DBP":
        return np.min(part)
    
def compute_all_features(part, feature_names):
    """
    Za posamezen part zračunamo vse featureje iz feature_names.
    Feature names je seznam vseh featurjev ki jih hočemo računati.

    """
    features = {}
    for feature_name in feature_names:
        features[feature_name] = compute_feature(part, feature_name)
    return features

def construct_X(data=data, feature_names=FEATURES):
    """
    data je seznam slovarjev, ki vsebujejo podatke o valovih in parametrih.
    feature_names je seznam vseh featurjev, ki jih hočemo računati.

    Vrne pandas dataframe, kjer so stolpci featureji in parametri C, R, S.
    """
    df = pd.DataFrame(columns=feature_names + ['C', 'R', 'S'])

    rows = []
    for elt in data:
        part = elt['part']
        params = elt["params"]

        features = compute_all_features(part, feature_names)
        features['C'] = params[0]
        features['R'] = params[1]
        features['S'] = params[2]

        features = dict((key, [features[key]]) for key in features)
        # print(features)

        row = pd.DataFrame(features)
        rows.append(row)
 
    df = pd.concat(rows, ignore_index=True)
    df.reset_index(inplace=True)

    df.to_excel("data/features.xlsx")
    df.to_csv("data/features.csv")
    
    return df

if __name__=="__main__":

    construct_X(data[:5], FEATURES)
        



