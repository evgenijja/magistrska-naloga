from simulation_code import *

"""
Here I will keep all the history on running simulations.

NO RUNNING SIMULATIONS OUTSIDE OF THIS SCRIPT I BEG YOU.
"""

DtSimulations = 8.5*3

# =================================================================================================================================================================
"""
Datum: 25. 4. 2024

Začnem z večjim prostorom za simulacije in večjim korakom: 
    contractility = np.arange(0.1, 2.5, 0.1)
    resistance = np.arange(0.1, 2.5, 0.1)
    compliance = np.arange(0.1, 2.5, 0.1)
    
Simuliram 2D prostor za fiksen C. Začnem z C = 0.1 in nato povečujem za 0.1.
        
Simulacije shranim v mapo: simulation_results/20240425/results_for_C_value.json

"""

resistance = np.arange(0.1, 2.5, 0.1)
compliance = np.arange(0.1, 2.5, 0.1)
C = 0.1

triples = []

for R in resistance:
    for S in compliance:
        triples.append([C, R, S])

# initialize json file { "C = 0.1" : []}
with open(f"simulation_results/20240425/results_for_C_{C}.json", 'w') as file:
    json.dump([], file, indent=4)

with open(f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", 'w') as file:
    json.dump([], file, indent=4)

run_multiple_simulations(triples, DtSimulations, f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", f"simulation_results/20240425/results_final_for_C_{C}.json")
