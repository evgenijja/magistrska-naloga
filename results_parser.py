import numpy as np
import pandas as pd

import json
import matplotlib.pyplot as plt

filepaths = [
    "simulation_results/20240425/results_final_for_C_0.1.json",
    "simulation_results/20240425/results_final_for_C_0.2.json",
    "simulation_results/20240425/results_final_for_C_0.3.json",
    "simulation_results/20240425/results_final_for_C_0.4.json",
    "simulation_results/20240425/results_final_for_C_0.5.json",
    "simulation_results/20240425/results_final_for_C_0.6.json",
    "simulation_results/20240425/results_final_for_C_0.7.json",
    "simulation_results/20240425/results_final_for_C_0.8.json",
    "simulation_results/20240425/results_final_for_C_0.9.json",
    "simulation_results/20240425/results_final_for_C_1.json",
    "simulation_results/20240425/results_final_for_C_1.1.json",
    "simulation_results/20240425/results_final_for_C_1.2.json",
    "simulation_results/20240425/results_final_for_C_1.3.json",
    "simulation_results/20240425/results_final_for_C_1.4.json",
    "simulation_results/20240425/results_final_for_C_1.5.json",
    "simulation_results/20240425/results_final_for_C_1.6.json",
    "simulation_results/20240425/results_final_for_C_1.7.json",
    "simulation_results/20240425/results_final_for_C_1.8.json",
    "simulation_results/20240425/results_final_for_C_1.9.json",
    "simulation_results/20240425/results_final_for_C_2.json",
    "simulation_results/20240425/results_final_for_C_2.1.json",
    "simulation_results/20240425/results_final_for_C_2.2.json",
    "simulation_results/20240425/results_final_for_C_2.3.json",
    "simulation_results/20240425/results_final_for_C_2.4.json",
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
        print(filepath)
        # naložimo podatke
        with open(filepath, "r") as json_file:
            data = json.load(json_file)


        for key in data:
            new_data = key_parser(key)
            part = data[key]['PART']
            if part == None or len(set(part)) == 1:
                part = 427 * [0]

                new_data['PART'] = part 
                results.append(new_data)

    return results

def visualise_3d(results):
    """
    Take the results and plot a 3d plot to show which points are already simulated or not. 
    """
    x, y, z = [], [], []
    # loop through results
    for result in results:
        x.append(result["C"])
        y.append(result["R"])
        z.append(result["S"])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel("C")
    ax.set_ylabel("R")
    ax.set_zlabel("S")
    plt.show()

    


if __name__=="__main__":

    results = get_results(filepaths)
    # visualise_3d(results)

    # write results to a new json file
    with open("data/results_final.json", 'w') as file:
        json.dump(results, file, indent=4)