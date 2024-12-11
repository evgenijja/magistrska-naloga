import numpy as np
import pandas as pd

from reading_data import *

from scipy.interpolate import CubicSpline


"""
Predpostavljamo, da dobimo tukaj val, na katerem ni bilo čisto nič narejeno. 

Narediti želimo dve transformaciji:
    1. Prvi del premaknemo na konec
        - to bi si želeli narediti malo bolj elegantno ampak problem je, da lahko to vpliva na vse značilke povezane s časom
        - mogoče bi jih morali vedno odrezati isto število?
    2. Val želimo zožat na x os od 0 do 1 
    3. Dodatno bi želeli eksperimentirati kaj se zgodi, če tudi y os skrčimo na 0 do 1 za tiste featurje, za katere je smiselno
"""

def transform_part(part, transformation=[]):
    """
    Izvedemo transformacijo na partu, ki je direktno dobljen od simulacij. 
    scale_x niti ne rabim delat, ker samo potem delim z dolžino x osi, da imam skalirano na [0,1]

    """
    for trans in transformation:
        if trans == "move_first_to_end":
            # TODO ali lahko nareidm to bolj elegantno?
            min_index = np.argmin(part[:300])
            first_part, second_part = part[min_index:], part[:min_index]
            diff = second_part[0] - first_part[-1]

            part = part[min_index:] + [part[:min_index][j] - diff for j in range(len(part[:min_index]))]
            # part = part[min_index:] + part[:min_index]

        # elif trans == "scale_x":
        #     scale = len(part)

        elif trans == "scale_y":
            # odštejem diastolni pritisk
            diastolic_pressure = np.min(part)
            part = [elt - diastolic_pressure for elt in part]
            # delimo z razliko med sistoličnim in diastoličnim pritiskom
            systolic_pressure = np.max(part)
            part = [elt / systolic_pressure for elt in part]

    return part, first_part

# create a list of feature_names that we want to compute based on the function below
feature_names = [
    "čas do SP", 
    "čas od SP do DN", 
    "čas od SP do konca", 
    "čas od začetka do DN", 
    "čas od DN do konca",
    "razlika med BP pri SP in BP pri DN",
    "razlika med BP pri DN in diastolnim pritiskom",
    "vrednost BP pri SP",
    "vrednost BP pri DN",
    "razmerje med tlakom pri DN in PP",
    "razmerje med razliko tlaka pri SP in DN in PP",
    "povprečje vala",
    "povprečje diastolnega dela vala",
    "povprečje sistolnega dviga",
    "popvrečje sistolnega padca",
    "povprečje vala kjer se val spušča",
    "povprečje vala kjer odštejemo diastolni tlak",
    "povprečje diastolnega dela vala kjer odštejemo diastolni tlak",
    "povprečje sistolnega dviga kjer odštejemo diastolni tlak",
    "popvrečje sistolnega padca kjer odštejemo diastolni tlak",
    "povprečje vala kjer se val spušča kjer odštejemo diastolni tlak",
    "PP",
    "SBP",
    "DBP",
    "MAP",
    "površina pod celim valom",
    "površina pod sistolnim delom vala",
    "površina pod valom med SP in DN",
    "površina pod diastolnim delom vala",
    "površina pod valom kjer se val spušča",
    "površina pod valom kjer odštejemo diastolni tlak",
    "površina pod sistolnim delom vala kjer odštejemo diastolni tlak",
    "površina pod valom med SP in DN kjer odštejemo diastolni tlak",
    "površina pod diastolnim delom vala kjer odštejemo diastolni tlak",
    "površina pod valom kjer se val spušča kjer odštejemo diastolni tlak",
    "maksimalni naklon",
    "maksimalna vrednost drugega odvoda",
    "PP / čas do SP",
    "(BP pri DN - BP pri SP) / čas od SP do DN",
    "diastolni tlak - tlak pri DN / čas od DN do konca",
    "standardna deviacija tlaka",
    "standardna deviacija tlaka z odštetim diastolnim tlakom",
    "skeweness (tretji moment) tlaka",
    "skeweness (tretji moment) tlaka z odštetim diastolnim tlakom",
    "kurtosis (četrti moment) tlaka",
    "kurtosis (četrti moment) tlaka z odštetim diastolnim tlakom",
    "SV",
    "CO"
]
                 


def compute_feature(part, feature_name, first_part):
    """
    Sprejmemo part, na katerem smo izvedli samo osnovno transformacijo. 
    """
    # systolic peak index
    systolic_peak_index = np.argmax(part)

    # dicrotic notch index = max second derivative after peak
    x = np.arange(len(first_part))
    # print(first_part )
    spl = CubicSpline(x, first_part)
    y_new = spl(x)
    deriv1 = spl(x, nu=1)
    deriv2 = spl(x, nu=2)

    # if first_part is not None:
    dicrotic_notch_index = np.argmax(deriv2[systolic_peak_index:]) + systolic_peak_index

    # značilke, ki morajo bit na navadnem valu zračunane (ne na tistem, kjer je y os skalirana na [0,1])
    if feature_name == "čas do SP":
        return systolic_peak_index / len(part)

    elif feature_name == "čas od SP do DN":
        return (dicrotic_notch_index - systolic_peak_index) / len(part)
    
    elif feature_name == "čas od SP do konca":
        return (len(part) - systolic_peak_index) / len(part)
    
    elif feature_name == "čas od začetka do DN":
        return dicrotic_notch_index / len(part)
    
    elif feature_name == "čas od DN do konca":
        return (len(part) - dicrotic_notch_index) / len(part)
    
    elif feature_name == "razlika med BP pri SP in BP pri DN":
        return part[systolic_peak_index] - part[dicrotic_notch_index]
    
    elif feature_name == "razlika med BP pri DN in diastolnim pritiskom":
        return part[dicrotic_notch_index] - np.min(part)
    
    elif feature_name == "vrednost BP pri SP":
        return part[systolic_peak_index]
    
    elif feature_name == "vrednost BP pri DN":
        return part[dicrotic_notch_index]
    
    elif feature_name == "razmerje med tlakom pri DN in PP":
        return part[dicrotic_notch_index] / (part[systolic_peak_index] - np.min(part))
    
    elif feature_name == "razmerje med razliko tlaka pri SP in DN in PP":
        return (part[systolic_peak_index] - part[dicrotic_notch_index]) / (part[systolic_peak_index] - np.min(part))
    
    elif feature_name == "povprečje vala":
        return np.mean(part)
    
    elif feature_name == "povprečje diastolnega dela vala":
        return np.mean(part[dicrotic_notch_index:])
    
    elif feature_name == "povprečje sistolnega dviga":
        return np.mean(part[:systolic_peak_index])
    
    elif feature_name == "popvrečje sistolnega padca":
        return np.mean(part[systolic_peak_index:dicrotic_notch_index])
    
    elif feature_name == "povprečje vala kjer se val spušča":
        return np.mean(part[systolic_peak_index:])
    
    elif feature_name == "povprečje vala kjer odštejemo diastolni tlak":
        diastolni_tlak = np.min(part)
        part = [elt - diastolni_tlak for elt in part]
        return np.mean(part)
    
    elif feature_name == "povprečje diastolnega dela vala kjer odštejemo diastolni tlak":
        diastolni_tlak = np.min(part)
        part = [elt - diastolni_tlak for elt in part]
        return np.mean(part[dicrotic_notch_index:])
    
    elif feature_name == "povprečje sistolnega dviga kjer odštejemo diastolni tlak":
        diastolni_tlak = np.min(part)
        part = [elt - diastolni_tlak for elt in part]
        return np.mean(part[:systolic_peak_index])
    
    elif feature_name == "popvrečje sistolnega padca kjer odštejemo diastolni tlak":
        diastolni_tlak = np.min(part)
        part = [elt - diastolni_tlak for elt in part]
        return np.mean(part[systolic_peak_index:dicrotic_notch_index])
    
    elif feature_name == "povprečje vala kjer se val spušča kjer odštejemo diastolni tlak":
        diastolni_tlak = np.min(part)
        part = [elt - diastolni_tlak for elt in part]
        return np.mean(part[systolic_peak_index:])
    
    elif feature_name == "PP":
        return part[systolic_peak_index] - np.min(part)
    
    elif feature_name == "SBP":
        return part[systolic_peak_index]
    
    elif feature_name == "DBP":
        return np.min(part)
    
    elif feature_name == "MAP":
        return 1/3 * (np.max(part) - np.min(part)) + np.min(part)

    elif feature_name == "površina pod celim valom":
        return np.trapz(part)
    
    elif feature_name == "površina pod sistolnim delom vala":
        return np.trapz(part[:systolic_peak_index])
    
    elif feature_name == "površina pod valom med SP in DN":
        return np.trapz(part[systolic_peak_index:dicrotic_notch_index])
    
    elif feature_name == "površina pod diastolnim delom vala":
        return np.trapz(part[dicrotic_notch_index:])
    
    elif feature_name == "površina pod valom kjer se val spušča":
        return np.trapz(part[systolic_peak_index:])
    
    elif feature_name == "površina pod valom kjer odštejemo diastolni tlak":
        diastolni_tlak = np.min(part)
        part = [elt - diastolni_tlak for elt in part]
        return np.trapz(part)   
    
    elif feature_name == "površina pod sistolnim delom vala kjer odštejemo diastolni tlak":
        diastolni_tlak = np.min(part)
        part = [elt - diastolni_tlak for elt in part]
        return np.trapz(part[:systolic_peak_index])
    
    elif feature_name == "površina pod valom med SP in DN kjer odštejemo diastolni tlak":
        diastolni_tlak = np.min(part)
        part = [elt - diastolni_tlak for elt in part]
        return np.trapz(part[systolic_peak_index:dicrotic_notch_index])
    
    elif feature_name == "površina pod diastolnim delom vala kjer odštejemo diastolni tlak":
        diastolni_tlak = np.min(part)
        part = [elt - diastolni_tlak for elt in part]
        return np.trapz(part[dicrotic_notch_index:])
    
    elif feature_name == "površina pod valom kjer se val spušča kjer odštejemo diastolni tlak":
        diastolni_tlak = np.min(part)
        part = [elt - diastolni_tlak for elt in part]
        return np.trapz(part[systolic_peak_index:])
    
    elif feature_name == "maksimalni naklon":
        return np.max(deriv1) 
    
    elif feature_name == "maksimalna vrednost drugega odvoda":
        return np.max(deriv2)
    
    elif feature_name == "PP / čas do SP":
        return (part[systolic_peak_index] - np.min(part)) / (systolic_peak_index / len(part))
    
    elif feature_name == "(BP pri DN - BP pri SP) / čas od SP do DN":
        return (part[dicrotic_notch_index] - part[systolic_peak_index]) / ((dicrotic_notch_index - systolic_peak_index) / len(part))
    

    elif feature_name == "diastolni tlak - tlak pri DN / čas od DN do konca":
        return (np.min(part) - part[dicrotic_notch_index]) / ((len(part) - dicrotic_notch_index) / len(part))
    
    elif feature_name == "standardna deviacija tlaka":
        return np.std(part)
    
    elif feature_name == "standardna deviacija tlaka z odštetim diastolnim tlakom":
        diastolni_tlak = np.min(part)
        part = [elt - diastolni_tlak for elt in part]
        return np.std(part)
    
    elif feature_name == "skeweness (tretji moment) tlaka":
        return pd.Series(part).skew()
    
    elif feature_name == "skeweness (tretji moment) tlaka z odštetim diastolnim tlakom":
        diastolni_tlak = np.min(part)
        part = [elt - diastolni_tlak for elt in part]
        return pd.Series(part).skew()
    
    elif feature_name == "kurtosis (četrti moment) tlaka":
        return pd.Series(part).kurt()
    
    elif feature_name == "kurtosis (četrti moment) tlaka z odštetim diastolnim tlakom":
        diastolni_tlak = np.min(part)
        part = [elt - diastolni_tlak for elt in part]
        return pd.Series(part).kurt()
    
    elif feature_name == "SV":
        return 1/3.5 * (part[systolic_peak_index] - np.min(part)) / (part[systolic_peak_index] + np.min(part)) * 1000
    
    elif feature_name == "CO":
        return (1/3.5 * (part[systolic_peak_index] - np.min(part)) / (part[systolic_peak_index] + np.min(part)) * 1000 / 1000) * 75
    

    

def generate_feature_matrix(data, feature_names):

    computed_features = []
    for point in data:
        params = point['params']
        C, R, S = params[0], params[1], params[2]

        part = point['part']
        part, first_part = transform_part(part, transformation=["move_first_to_end"])
        # print(len(part))
        computed_features_point = [C, R, S]
        print(C, R, S)
        for feature in feature_names:
            computed_features_point.append(compute_feature(part, feature, first_part))
        
        computed_features.append(computed_features_point)

    df = pd.DataFrame(computed_features, columns=["C", "R", "S"] + feature_names)
    df.to_csv("data/features_FINAL.csv")
    df.to_excel("data/features_FINAL.xlsx")
    return df

# 1.9500000000000013 1.35 0.1

def read_parsed_data(filter=True):

    with open("data/parsed_data2.json", "r") as json_file:
        data = json.load(json_file)

    print("Število VSEH simulacij:", len(data))

    print("Filtriram ...")

    print("Odstranim tiste, ki imajo kak nan")

    new_data = []
    for point in data:
        part = point['part']
        if not np.isnan(part).any() and part != len(part) * [0] and len(set(part)) > 1 and sum([0 if elt >= 0 else 1 for elt in part]) == 0:
            if point not in new_data:
                new_data.append(point)

    print("Število simulacij po filtriranju:", len(new_data))

    return new_data

if __name__=="__main__":

    data = read_parsed_data()
    df = generate_feature_matrix(data, feature_names) # 20:23

    # for point in data:
    #     params = point['params']
    #     C, R, S = round(params[0], 2), round(params[1], 2), round(params[2], 2)
    #     if C == 2 and R == 0.7 and S == 0.15:
    #         print(point['part'])


