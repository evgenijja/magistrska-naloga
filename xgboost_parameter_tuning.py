from magistrska_koda.xgboost_implementation_first import *
import json 

"""
Želimo najti optimalne parametre za naš model.

Definiramo prostor parametrov, ki ga bomo raziskovali.
"""

def load_params():
    with open('params.json', 'r') as file:
        parameter_combinations = json.load(file)
    return parameter_combinations

def try_params(loaded_params, X, C, R, S, title='contractility', mode='cv', k=10):

    final_results = []

    for i, params in enumerate(loaded_params):

        print(f"Parameter combination number {i + 1} out of {len(loaded_params)}")

        p = {
            "objective": "reg:squarederror",
            "max_depth": params['max_depth'],
            "subsample": params['subsample'],
            "eta": params['eta'],
            "eval_metric": "rmse",
            "seed": 42
        }

        early_stopping_rounds = params['early_stopping_rounds']
        num_round = params['num_boost_round']

        results = xgboost_cv(p, early_stopping_rounds, num_round, X, C, R, S, title=title, mode=mode, k=k)
        # results = {1 : 'test'}

        final_results.append(results)

        file_path = f"xgboost_plots/{title}_parameter_results_11092024.json"

        with open(file_path, 'w') as file:
            json.dump(final_results, file, indent=4)


    return final_results

def find_optimal_params(filepath):

    with open(filepath, 'r') as file:
        results = json.load(file)

    sorted_results = sorted(
        results,
        key=lambda x: (x['mean_r2'], -x['std_r2']),
        reverse=True
    )

    # write new file with sorted results
    with open(filepath, 'w') as file:
        json.dump(sorted_results, file, indent=4)

    return sorted_results[0]

if __name__ == "__main__":

    params = load_params()

    X, C, R, S, feature_names = load_new_data(filepath="data/features_11092024.xlsx")

    start_time = time.time()
    start_ctime = time.ctime()
    print(f"Starting at {start_ctime}")

    results = try_params(params, X, C, R, S, title='resistance', mode='cv', k=10)
    results = try_params(params, X, C, R, S, title='contractility', mode='cv', k=10)
    results = try_params(params, X, C, R, S, title='stiffness', mode='cv', k=10)

    # end_time = time.time()
    # end_ctime = time.ctime()

    # print(f"Started at {start_ctime}")
    # print(f"Finished at {end_ctime}")
    # print(f"Elapsed time: {end_time - start_time} seconds")

    filepath_c = f"xgboost_plots/contractility_parameter_results_11092024.json"
    filepath_r = f"xgboost_plots/resistance_parameter_results_11092024.json"
    filepath_s = f"xgboost_plots/stiffness_parameter_results_11092024.json"

    best_params_c = find_optimal_params(filepath_c)
    best_params_r = find_optimal_params(filepath_r)
    best_params_s = find_optimal_params(filepath_s)

    print("Contractility best params:")
    print("Eta: ", best_params_c['eta'])
    print("Max depth: ", best_params_c['max_depth'])
    print("Subsample: ", best_params_c['subsample'])
    print("Num boost rounds: ", best_params_c['num_round'])
    print("Early stopping rounds: ", best_params_c['early_stopping_rounds'])
    print("=============================================================================")

    print("Resistance best params:")
    print("Eta: ", best_params_r['eta'])
    print("Max depth: ", best_params_r['max_depth'])
    print("Subsample: ", best_params_r['subsample'])
    print("Num boost rounds: ", best_params_r['num_round'])
    print("Early stopping rounds: ", best_params_r['early_stopping_rounds'])
    print("=============================================================================")

    print("Stiffness best params:")
    print("Eta: ", best_params_s['eta'])
    print("Max depth: ", best_params_s['max_depth'])
    print("Subsample: ", best_params_s['subsample'])
    print("Num boost rounds: ", best_params_s['num_round'])
    print("Early stopping rounds: ", best_params_s['early_stopping_rounds'])

    # Contractility best params:
    # Eta:  0.08
    # Max depth:  6
    # Subsample:  1
    # Num boost rounds:  1200
    # Early stopping rounds:  90
    # =============================================================================
    # Resistance best params:
    # Eta:  0.1
    # Max depth:  5
    # Subsample:  1
    # Num boost rounds:  700
    # Early stopping rounds:  60
    # =============================================================================
    # Stiffness best params:
    # Eta:  0.01
    # Max depth:  8
    # Subsample:  1
    # Num boost rounds:  5000
    # Early stopping rounds:  100

    # new

#     Contractility best params:
# Eta:  0.05
# Max depth:  6
# Subsample:  1
# Num boost rounds:  1000
# Early stopping rounds:  50
# =============================================================================
# Resistance best params:
# Eta:  0.05
# Max depth:  6
# Subsample:  1
# Num boost rounds:  1000
# Early stopping rounds:  50
# =============================================================================
# Stiffness best params:
# Eta:  0.08
# Max depth:  8
# Subsample:  1
# Num boost rounds:  1400
# Early stopping rounds:  100




