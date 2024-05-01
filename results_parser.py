import numpy as np
import pandas as pd

import json

filepaths = [
    "simulation_results/20240425/results_final_for_C_0.1.json",
    # "simulation_results/20240425/results_final_for_C_0.2.json",
    "simulation_results/20240425/results_final_for_C_0.3.json",
    # "simulation_results/20240425/results_final_for_C_0.4.json",
    "simulation_results/20240425/results_final_for_C_0.5.json",
    # "simulation_results/20240425/results_final_for_C_0.6.json",
    "simulation_results/20240425/results_final_for_C_0.7.json",
    # "simulation_results/20240425/results_final_for_C_0.8.json",
    "simulation_results/20240425/results_final_for_C_0.9.json",
    # "simulation_results/20240425/results_final_for_C_1.0.json"
]

def key_parser(key):
    """
    Ključi so oblike "C = 0.1, R = 0.1, S = 0.1"
    Hočemo jih tako sparsat, da dobim ven vrednosti za vsak parameter.
    """
    parts = key.split(", ")  # Split by comma and space
    parsed_dict = {}
    for part in parts:
        key, value = part.split(" = ")  # Split by space and equals sign
        parsed_dict[key.strip()] = float(value)  # Convert value to float and strip extra spaces
    return parsed_dict


def get_results(filepaths):
    """
    Vrne seznam seznamov 
    """

    results = []

    for filepath in filepaths:

        # naložimo podatke
        with open(filepath, "r") as json_file:
            data = json.load(json_file)


        for key in data:
            new_data = key_parser(key)
            part = data[key]['PART']
            if len(set(part)) == 1:
                part = 427 * [0]

            new_data['PART'] = part 
            results.append(new_data)

    return results

if __name__=="__main__":

    results = get_results(filepaths)

    # write results to a new json file
    with open("data/results_new.json", 'w') as file:
        json.dump(results, file, indent=4)