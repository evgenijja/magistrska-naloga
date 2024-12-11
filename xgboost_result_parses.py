import json 

filepaths = [
    "xgboost_plots/contractility_parameter_results_11092024.json",
    "xgboost_plots/resistance_parameter_results_11092024.json",
    "xgboost_plots/stiffness_parameter_results_11092024.json"
]

if __name__=="__main__":

    all_results = []
    for filepath in filepaths:
        with open(filepath, 'r') as f:
            results = json.load(f)
            all_results.append(results)

    new_json = []
    for i in range(len(all_results[0])):
        print(i)
        max_depth = all_results[0][i]['max_depth']
        eta = all_results[0][i]['eta']
        num_round = all_results[0][i]['num_round']
        early_stopping_rounds = all_results[0][i]['early_stopping_rounds']

        for j in range(len(all_results[1])):
            if all_results[1][j]['max_depth'] == max_depth and all_results[1][j]['eta'] == eta and all_results[1][j]['num_round'] == num_round and all_results[1][j]['early_stopping_rounds'] == early_stopping_rounds:
                p1 = all_results[1][j]
                print("good")
                break

        for j in range(len(all_results[2])):
            if all_results[2][j]['max_depth'] == max_depth and all_results[2][j]['eta'] == eta and all_results[2][j]['num_round'] == num_round and all_results[2][j]['early_stopping_rounds'] == early_stopping_rounds:
                # p2 = all_results[1][j]
                print("good")
                p2 = all_results[2][j]
                break
        res = {
            'max_depth' : all_results[0][i]['max_depth'], 
            'eta' : all_results[0][i]['eta'],
            'num_boost_round' : all_results[0][i]['num_round'],
            'early_stopping_rounds' : all_results[0][i]['early_stopping_rounds'],
            'mean_r2_C' : round(all_results[0][i]['mean_r2'], 3),
            'mean_r2_R' : round(p1['mean_r2'], 3),
            'mean_r2_S' : round(p2['mean_r2'], 3),
            'avg' : (round(all_results[0][i]['mean_r2'], 3) + round(p1['mean_r2'], 3) + round(p2['mean_r2'], 3)) / 3,
            # 'std_r2_C' : round(all_results[0][i]['std_r2'], 3),
            # 'std_r2_R' :    round(p1['std_r2'], 3),
            # 'std_r2_S' : round(p2['std_r2'], 3),
        }
        new_json.append(res)

    # order new_json by avg
    new_json = sorted(new_json, key=lambda x: x['avg'], reverse=True)

    with open("xgboost_plots/all_results_20241209.json", 'w') as f:
        json.dump(new_json, f, indent=4)

