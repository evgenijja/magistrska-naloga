import matlab.engine
import numpy as np

import os
import json

import time
import multiprocessing

from scipy.io import loadmat

def update_json(filename, new_key, new_data):
    """
    Update json file with new info. If file does not exist, create it and add new info.
    We need this for mid-simulation saving of results.
    """
    
    if os.path.exists(filename):
        with open(filename, 'r+') as file:
            results_dict = json.load(file)
            to_insert = {new_key: new_data}
            results_dict.append(to_insert)
            file.seek(0)
            json.dump(results_dict, file, indent=4) 
            # 
            # results_dict.update(to_insert)

    else:
        results_dict = {}
        results_dict[new_key] = new_data
        with open(filename, 'w') as file:
            json.dump(results_dict, file, indent=4)


def run_one_simulation(eng, C, R, S, DtSimulations, code_path="CircAdapt_Niklas\CircAdapt_Niklas"):
    """
    Runs function in matlab that simulates the data.
    Input parameters are C, R, S for contractility, resistance and compliance.

    At the end I want to save values for PART, FLOW, CAV and also P.mat and the SVR matrix.
    """

    C, R, S = float(C), float(R), float(S)

    print("Running simulation for C={}, R={}, S={}".format(C, R, S))

    start_time = time.time()

    eng.addpath(code_path)

    part, flow, cav, diff = eng.eval(f"runSimulation({C}, {R}, {S}, {DtSimulations})", nargout=4)

    part = list(np.array(part).flatten().tolist())
    part = list(map(lambda x: x if type(x) != complex else -1, part))

    flow = list([list(elt) for elt in flow])
    flow = list(np.transpose(np.array(flow)))
    flow = list([list(elt) for elt in flow])

    cav = list([list(elt) for elt in cav])
    cav = list(np.transpose(np.array(cav)))
    cav = list([list(elt) for elt in cav])

    stop_time = time.time()

    print("Simulation took {} seconds.".format(stop_time - start_time))

    if len(part) == 1:
        return {"PART": None, "FLOW": None, "CAV": None, "DIFF": None, "time": None}

    return {"PART": part, "FLOW": flow, "CAV": cav, "DIFF": diff, "time": stop_time - start_time}


def run_multiple_simulations(triples, DtSimulation, filename, filename_final):
    """
    TODO - pazi da se ti kateri triple ne ponovi
    """
    eng = matlab.engine.start_matlab()

    results_dict = {}

    for triple in triples:
        C, R, S = triple

        # predpostavljam, da se noben triple ne ponovi
        result = run_one_simulation(eng, C, R, S, DtSimulation)

        results_dict[f"C = {C}, R = {R}, S = {S}"] = result
        # print(results_dict)

        update_json(filename, f"C = {C}, R = {R}, S = {S}", result)

    eng.quit()

    with open(filename_final, 'w') as file:
        json.dump(results_dict, file, indent=4)

    return results_dict



if __name__=="__main__":


    eng = matlab.engine.start_matlab()
    run_one_simulation(eng, 2.5, 2.5, 2.5, 3)
    eng.quit()

    # triples = [(0.9, 0.9, 0.9), (0.8, 0.8, 0.8), (0.7, 0.7, 0.7)]
    # res = run_multiple_simulations(triples, "simulation_results/results.json", "simulation_results/results_final.json")