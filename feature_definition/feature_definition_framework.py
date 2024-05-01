import numpy as np
import pandas as pd

from reading_data import *

def area_under_curve(part):
    area = 0.0
    n = len(part)

    for i in range(1, n):
        # Trapezoidal rule formula
        area += np.min([part[i-1], part[i]]) + (np.max([part[i-1], part[i]]) - np.min([part[i-1], part[i]])) * 0.5

    return area


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

def compute_feature(part, feature_name):
    """
    Za posamezen part zračunamo nek feature.
    """
    dicrotic_notch_index = len(part) // 3
     
    if feature_name == "MAP":
        feature_value = np.mean(part)
    elif feature_name == "PP":
        feature_value = np.max(part) - np.min(part)
    elif feature_name == "SBP":
        feature_value = np.max(part)
    elif feature_name == "DBP":
        feature_value = np.min(part)
    
    elif feature_name == "flow":
        pass # TODO

    
    elif feature_name == 'rDNP':
        feature_value = part[dicrotic_notch_index] - np.min(part)

    elif feature_name == 'DP': # descending pressure
        feature_value = np.max(part) - part[dicrotic_notch_index]

    # elif feature_name == 'DNIx':
    #     feature_value = (part[dicrotic_notch_index] - np.min(part)) / (np.max(part) - np.min(part)) * 100

    # elif feature_name == 'DIx':
    #     feature_value =  (np.max(part) - part[dicrotic_notch_index]) / (np.max(part) - np.min(part)) * 100

    elif feature_name == 't_sys':
        systolic_peak_index = np.argmax(part)


    elif feature_name == 'total_area' or feature_name == 'systolic_area' or feature_name == 'diastolic_area':
        total_area = area_under_curve(part)

        systolic_peak = np.argmax(part)
        systolic_part = part[:systolic_peak]

        systolic_area = area_under_curve(systolic_part)
        diastolic_area = total_area - systolic_area

        if feature_name == 'total_area':
            feature_value = total_area
        elif feature_name == 'systolic_area':
            feature_value = systolic_area
        elif feature_name == 'diastolic_area':
            feature_value = diastolic_area

    elif feature_name == 'relative_total_area' or feature_name == 'relative_systolic_area' or feature_name == 'relative_diastolic_area':
        part = list(map(lambda x: x - np.min(part), part))
        total_area = area_under_curve(part)

        systolic_peak = np.argmax(part)
        systolic_part = part[:systolic_peak]

        systolic_area = area_under_curve(systolic_part)
        diastolic_area = total_area - systolic_area

        if feature_name == 'relative_total_area':
            feature_value = total_area
        elif feature_name == 'relative_systolic_area':
            feature_value = systolic_area
        elif feature_name == 'relative_diastolic_area':
            feature_value = diastolic_area

    elif feature_name == 'stroke_volume':
        feature_value = 1/3.5 * ((np.max(part) - np.min(part))/(np.max(part) + np.min(part))) * 1000

    elif feature_name == 'cardiac_output':
        stroke_volume = 1/3.5 * ((np.max(part) - np.min(part))/(np.max(part) + np.min(part))) * 1000
        feature_value = (stroke_volume / 1000) * 72

    elif feature_name == 'duration_systole' or feature_name == 'duration_diastole':
        systolic_peak = np.argmax(part)
        if feature_name == 'duration_systole':
            feature_value = systolic_peak / len(part) * 0.85 # heartbeat
        if feature_name == 'duration_diastole':
            feature_value = 0.85 - systolic_peak / len(part) * 0.85

    elif feature_name == 'cven':
        feature_value = part[120]

    # SLOPES
    elif feature_name == "systolic_upstroke_slope":
        feature_value = (np.max(part) - np.min(part)) / (np.argmax(part) / len(part) * 0.85)

    elif feature_name == "systolic_downstroke_slope":
        feature_value = (part[dicrotic_notch_index] - np.max(part))

    # elif feature_name == "maximum slope":
    #     feature_value = None
        
    elif feature_name == 'diff_sbp':
        MAP = np.min(part) + 1/3*(np.max(part) - np.min(part))
        part = list(map(lambda x : x - MAP, part))
        feature_value = np.max(part)

    elif feature_name == 'diff_dbp':
        MAP = np.min(part) + 1/3*(np.max(part) - np.min(part))
        part = list(map(lambda x : x - MAP, part))
        feature_value = np.min(part)

    # elif feature_name == 'residuum':
    #     normal_part = normal_state()
    #     feature_value = part[120] - normal_part[120]


    return feature_value

def compute_area_features():
    """
    Značilke povezane s površino računamo posebej, da ne računamo površine vsakič znova.
    """
    
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
    # df = pd.DataFrame(columns=feature_names + ['C', 'R', 'S'])

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

    construct_X(data, FEATURES)
        



