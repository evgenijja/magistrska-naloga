import numpy as np
import pandas as pd

def load_new_data (filepath="data/new_features.xlsx"):

    # read dataframe that has and index column and header
    df = pd.read_excel(filepath, header=0, index_col=0)
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

renaming = {
    'čas do SP': 't_sp',
    'čas od SP do DN': 't_sp_dn',
    'čas od SP do konca': 't_sp_end',
    'čas od začetka do DN': 't_start_dn',
    'čas od DN do konca': 't_dn_end',
    'razlika med BP pri SP in BP pri DN': 'bpsp_bpdn_diff',
    'razlika med BP pri DN in diastolnim pritiskom': 'bpdn_d_diff',
    'vrednost BP pri SP': 'bpsp',
    'vrednost BP pri DN': 'bpdn',
    'razmerje med tlakom pri DN in PP': 'bpdn_pp_ratio',
    'razmerje med razliko tlaka pri SP in DN in PP': 'bpsp_bpdn_diff_pp_ratio',
    'povprečje vala': 'mean_bp',
    'povprečje diastolnega dela vala': 'mean_dn_end',
    'povprečje sistolnega dviga': 'mean_start_sp',
    'popvrečje sistolnega padca': 'mean_sp_dn',
    'povprečje vala kjer se val spušča': 'mean_sp_end',
    'povprečje vala kjer odštejemo diastolni tlak': 'mean_bp_wo_d',
    'povprečje diastolnega dela vala kjer odštejemo diastolni tlak': 'mean_dn_end_wo_d',
    'povprečje sistolnega dviga kjer odštejemo diastolni tlak': 'mean_start_sp_wo_d',
    'popvrečje sistolnega padca kjer odštejemo diastolni tlak': 'mean_sp_dn_wo_d',
    'povprečje vala kjer se val spušča kjer odštejemo diastolni tlak': 'mean_sp_end_wo_d',
    'PP': 'pp',
    'SBP': 'sbp',
    'DBP': 'dbp',
    'MAP': 'map',
    'površina pod celim valom': 'area',
    'površina pod sistolnim delom vala': 'area_start_sp',
    'površina pod valom med SP in DN': 'area_sp_dn',
    'površina pod diastolnim delom vala': 'area_dn_end',
    'površina pod valom kjer se val spušča': 'area_sp_end',
    'površina pod valom kjer odštejemo diastolni tlak': 'area_wo_d',
    'površina pod sistolnim delom vala kjer odštejemo diastolni tlak': 'area_start_sp_wo_d',
    'površina pod valom med SP in DN kjer odštejemo diastolni tlak': 'area_sp_dn_wo_d',
    'površina pod diastolnim delom vala kjer odštejemo diastolni tlak': 'area_dn_end_wo_d',
    'površina pod valom kjer se val spušča kjer odštejemo diastolni tlak': 'area_sp_end_wo_d',
    'maksimalni naklon': 'max_slope',
    'maksimalna vrednost drugega odvoda': 'max_2nd_derivative',
    'PP / čas do SP': 'pp_t_sp_ratio',
    '(BP pri DN - BP pri SP) / čas od SP do DN': 'bpsp_bpdn_t_sp_dn_ratio',
    'diastolni tlak - tlak pri DN / čas od DN do konca': 'd_bpdn_t_dn_end_ratio',
    'standardna deviacija tlaka': 'std_bp',
    'standardna deviacija tlaka z odštetim diastolnim tlakom': 'std_bp_wo_d',
    'skeweness (tretji moment) tlaka': 'skewness_bp',
    'skeweness (tretji moment) tlaka z odštetim diastolnim tlakom': 'skewness_bp_wo_d',
    'kurtosis (četrti moment) tlaka': 'kurtois_bp',
    'kurtosis (četrti moment) tlaka z odštetim diastolnim tlakom': 'kurtosis_bp_wo_d',
    'SV': 'sv',
    'CO': 'co'
}

if __name__ == "__main__":

    # X, C, R, S, feature_names = load_new_data()
    # print(dict((X.columns[i], '') for i in range(len(X.columns))))

    for i, elt in enumerate(renaming):
        print(f"{renaming[elt]}: {i}")