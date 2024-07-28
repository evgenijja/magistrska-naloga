from simulation_code import *

import time
import datetime

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

# resistance = np.arange(0.1, 2.5, 0.1)
# compliance = np.arange(0.1, 2.5, 0.1)
# C = 0.1 
# # DtSimulations = 8.5*3

# triples = []

# for R in resistance:
#     for S in compliance:
#         triples.append([C, R, S])

# # initialize json file { "C = 0.1" : []}
# with open(f"simulation_results/20240425/results_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# with open(f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# run_multiple_simulations(triples, DtSimulations, f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", f"simulation_results/20240425/results_final_for_C_{C}.json")


"""
Datum: 26. 4. 2024

Preskočim C = 0.2 in grem na C = 0.3.
"""

# resistance = np.arange(0.1, 2.5, 0.1)
# compliance = np.arange(0.1, 2.5, 0.1)
# C = 0.3 
# # DtSimulations = 8.5*3

# triples = []

# for R in resistance:
#     for S in compliance:
#         triples.append([C, R, S])

# # initialize json file { "C = 0.1" : []}
# with open(f"simulation_results/20240425/results_final_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# with open(f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# run_multiple_simulations(triples, DtSimulations, f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", f"simulation_results/20240425/results_final_for_C_{C}.json")

"""
Datum: 27. 4. 
"""

# resistance = np.arange(0.1, 2.5, 0.1)
# compliance = np.arange(0.1, 2.5, 0.1)
# C = 0.5
# # DtSimulations = 8.5*3

# triples = []

# for R in resistance:
#     for S in compliance:
#         triples.append([C, R, S])

# # initialize json file { "C = 0.1" : []}
# with open(f"simulation_results/20240425/results_final_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# with open(f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# print("Starting simulations for C = 0.5")
# print(datetime.datetime.now())
# # 2024-04-27 23:14:40.155363
# start_time = datetime.datetime.now()
# run_multiple_simulations(triples, DtSimulations, f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", f"simulation_results/20240425/results_final_for_C_{C}.json")
# print("Finished simulations for C = 0.5")
# print(datetime.datetime.now())
# end_time = datetime.datetime.now()

"""
Datum 28. 4. 2024
"""

# resistance = np.arange(0.1, 2.5, 0.1)
# compliance = np.arange(0.1, 2.5, 0.1)
# C = 0.7
# # DtSimulations = 8.5*3

# triples = []

# for R in resistance:
#     for S in compliance:
#         triples.append([C, R, S])

# # initialize json file { "C = 0.1" : []}
# with open(f"simulation_results/20240425/results_final_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# with open(f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# print("Starting simulations for C = 0.5")
# print(datetime.datetime.now())
# # 2024-04-28 10:03:35.175516
# start_time = datetime.datetime.now()
# run_multiple_simulations(triples, DtSimulations, f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", f"simulation_results/20240425/results_final_for_C_{C}.json")
# print("Finished simulations for C = 0.5")
# print(datetime.datetime.now())
# end_time = datetime.datetime.now()

"""
Datum: 30. 4. 2024
"""
# resistance = np.arange(0.1, 2.5, 0.1)
# compliance = np.arange(0.1, 2.5, 0.1)
# C = 0.9
# # DtSimulations = 8.5*3

# triples = []

# for R in resistance:
#     for S in compliance:
#         triples.append([C, R, S])

# # initialize json file { "C = 0.1" : []}
# with open(f"simulation_results/20240425/results_final_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# with open(f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# print(f"Starting simulations for C = {C}")
# print(datetime.datetime.now())
# # 2024-04-30 12:38:01.090698
# start_time = datetime.datetime.now()
# run_multiple_simulations(triples, DtSimulations, f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", f"simulation_results/20240425/results_final_for_C_{C}.json")
# print(f"Finished simulations for C = {C}")
# print(datetime.datetime.now())
# end_time = datetime.datetime.now()


"""
Datum 1. 5. 2024

"""
# resistance = np.arange(0.1, 2.5, 0.1)
# compliance = np.arange(0.1, 2.5, 0.1)
# C = 1
# DtSimulations = 8.5*3

# triples = []

# for R in resistance:
#     for S in compliance:
#         if (R, S) not in [(0.1, 0.1), (0.2, 0.2), (0.2, 0.1), (0.2, 0.1, 0.3, 0.1), (0.4, 0.1), (0.1, 0.2)]:
#             triples.append([C, R, S])

# # initialize json file { "C = 0.1" : []}
# with open(f"simulation_results/20240425/results_final_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# with open(f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# print(f"Starting simulations for C = {C}")
# print(datetime.datetime.now())
# # 2024-05-01 10:37:24.176685
# start_time = datetime.datetime.now()
# run_multiple_simulations(triples, DtSimulations, f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", f"simulation_results/20240425/results_final_for_C_{C}.json")
# print(f"Finished simulations for C = {C}")
# print(datetime.datetime.now())
# end_time = datetime.datetime.now()

# resistance = np.arange(0.1, 2.5, 0.1)
# compliance = np.arange(0.1, 2.5, 0.1)
# C = 1.1
# DtSimulations = 8.5*3

# triples = []

# for R in resistance:
#     for S in compliance:
#         if (R, S) not in [(0.1, 0.1), (0.2, 0.2), (0.2, 0.1),  (0.3, 0.1), (0.4, 0.1), (0.1, 0.2), (0.5, 0.1), (0.3, 0.2)]:
#             triples.append([C, R, S])

# # initialize json file { "C = 0.1" : []}
# with open(f"simulation_results/20240425/results_final_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# with open(f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# print(f"Starting simulations for C = {C}")
# print(datetime.datetime.now())
# # 2024-05-01 17:24:46.184413
# start_time = datetime.datetime.now()
# run_multiple_simulations(triples, DtSimulations, f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", f"simulation_results/20240425/results_final_for_C_{C}.json")
# print(f"Finished simulations for C = {C}")
# print(datetime.datetime.now())
# end_time = datetime.datetime.now()

# resistance = np.arange(0.1, 2.5, 0.1)
# compliance = np.arange(0.1, 2.5, 0.1)
# C = 1.3
# DtSimulations = 8.5*3

# triples = []

# for R in resistance:
#     for S in compliance:
#         if (R, S) not in [(0.1, 0.1), (0.2, 0.2), (0.2, 0.1),  (0.3, 0.1), (0.4, 0.1), (0.1, 0.2), (0.3, 0.2)]:
#             triples.append([C, R, S])

# # initialize json file { "C = 0.1" : []}
# with open(f"simulation_results/20240425/results_final_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# with open(f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# print(f"Starting simulations for C = {C}")
# print(datetime.datetime.now())
# # 2024-05-01 23:34:11.873159
# start_time = datetime.datetime.now()
# run_multiple_simulations(triples, DtSimulations, f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", f"simulation_results/20240425/results_final_for_C_{C}.json")
# print(f"Finished simulations for C = {C}")
# print(datetime.datetime.now())
# end_time = datetime.datetime.now()


"""
2. 5. 2024
"""
# resistance = np.arange(0.1, 2.5, 0.1)
# compliance = np.arange(0.1, 2.5, 0.1)
# C = 1.5
# DtSimulations = 8.5*3

# triples = []

# for R in resistance:
#     for S in compliance:
#         if (R, S) not in [(0.1, 0.1), (0.2, 0.2), (0.2, 0.1),  (0.3, 0.1), (0.4, 0.1), (0.1, 0.2)]:
#             triples.append([C, R, S])

# # initialize json file { "C = 0.1" : []}
# with open(f"simulation_results/20240425/results_final_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# with open(f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# print(f"Starting simulations for C = {C}")
# print(datetime.datetime.now())
# # 2024-05-02 11:27:55.695999
# start_time = datetime.datetime.now()
# run_multiple_simulations(triples, DtSimulations, f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", f"simulation_results/20240425/results_final_for_C_{C}.json")
# print(f"Finished simulations for C = {C}")
# print(datetime.datetime.now())
# end_time = datetime.datetime.now()

# resistance = np.arange(0.1, 2.5, 0.1)
# compliance = np.arange(0.1, 2.5, 0.1)
# C = 1.7
# DtSimulations = 8.5*3

# triples = []

# for R in resistance:
#     for S in compliance:
#         if (R, S) not in [(0.1, 0.1), (0.2, 0.2), (0.2, 0.1),  (0.3, 0.1), (0.4, 0.1), (0.1, 0.2)]:
#             triples.append([C, R, S])

# # initialize json file { "C = 0.1" : []}
# with open(f"simulation_results/20240425/results_final_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# with open(f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# print(f"Starting simulations for C = {C}")
# print(datetime.datetime.now())
# # 2024-05-02 17:29:23.353186
# start_time = datetime.datetime.now()
# run_multiple_simulations(triples, DtSimulations, f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", f"simulation_results/20240425/results_final_for_C_{C}.json")
# print(f"Finished simulations for C = {C}")
# print(datetime.datetime.now())
# end_time = datetime.datetime.now()

# resistance = np.arange(0.1, 2.5, 0.1)
# compliance = np.arange(0.1, 2.5, 0.1)
# C = 1.9
# DtSimulations = 8.5*3

# triples = []

# for R in resistance:
#     for S in compliance:
#         if (R, S) not in [(0.1, 0.1), (0.2, 0.2), (0.2, 0.1),  (0.3, 0.1), (0.4, 0.1), (0.1, 0.2)]:
#             triples.append([C, R, S])

# # initialize json file { "C = 0.1" : []}
# with open(f"simulation_results/20240425/results_final_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# with open(f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# print(f"Starting simulations for C = {C}")
# print(datetime.datetime.now())
# # 2024-05-02 17:29:23.353186
# start_time = datetime.datetime.now()
# run_multiple_simulations(triples, DtSimulations, f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", f"simulation_results/20240425/results_final_for_C_{C}.json")
# print(f"Finished simulations for C = {C}")
# print(datetime.datetime.now())
# end_time = datetime.datetime.now()

"""
5. 5. 2024
"""

# resistance = np.arange(0.1, 2.5, 0.1)
# compliance = np.arange(0.1, 2.5, 0.1)
# C = 2.1
# DtSimulations = 8.5*3

# triples = []

# for R in resistance:
#     for S in compliance:
#         if (R, S) not in [(0.1, 0.1), (0.2, 0.2), (0.2, 0.1),  (0.3, 0.1), (0.4, 0.1), (0.1, 0.2)]:
#             triples.append([C, R, S])

# # initialize json file { "C = 0.1" : []}
# with open(f"simulation_results/20240425/results_final_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# with open(f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# print(f"Starting simulations for C = {C}")
# print(datetime.datetime.now())
# # 2024-05-05 10:59:01.747272
# start_time = datetime.datetime.now()
# run_multiple_simulations(triples, DtSimulations, f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", f"simulation_results/20240425/results_final_for_C_{C}.json")
# print(f"Finished simulations for C = {C}")
# print(datetime.datetime.now())
# end_time = datetime.datetime.now()

# resistance = np.arange(0.1, 2.5, 0.1)
# compliance = np.arange(0.1, 2.5, 0.1)
# C = 2.3
# DtSimulations = 8.5*3

# triples = []

# for R in resistance:
#     for S in compliance:
#         if (R, S) not in [(0.1, 0.1), (0.2, 0.2), (0.2, 0.1),  (0.3, 0.1), (0.4, 0.1), (0.1, 0.2)]:
#             triples.append([C, R, S])

# # initialize json file { "C = 0.1" : []}
# with open(f"simulation_results/20240425/results_final_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# with open(f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# print(f"Starting simulations for C = {C}")
# print(datetime.datetime.now())
# # 2024-05-05 10:59:01.747272
# start_time = datetime.datetime.now()
# run_multiple_simulations(triples, DtSimulations, f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", f"simulation_results/20240425/results_final_for_C_{C}.json")
# print(f"Finished simulations for C = {C}")
# print(datetime.datetime.now())
# end_time = datetime.datetime.now()


# resistance = np.arange(0.1, 2.5, 0.1)
# compliance = np.arange(0.1, 2.5, 0.1)
# C = 1.2
# DtSimulations = 8.5*3

# triples = []

# for R in resistance:
#     for S in compliance:
#         if (round(R, 2), round(S, 2)) not in [(0.1, 0.1), (0.2, 0.2), (0.2, 0.1),  (0.3, 0.1), (0.4, 0.1), (0.5, 0.1), (0.1, 0.2), (0.1, 0.3)]:
#             triples.append([C, R, S])

# # initialize json file { "C = 0.1" : []}
# with open(f"simulation_results/20240425/results_final_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# with open(f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# print(f"Starting simulations for C = {C}")
# print(datetime.datetime.now())
# # 2024-05-05 10:59:01.747272
# start_time = datetime.datetime.now()
# run_multiple_simulations(triples, DtSimulations, f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", f"simulation_results/20240425/results_final_for_C_{C}.json")
# print(f"Finished simulations for C = {C}")
# print(datetime.datetime.now())
# end_time = datetime.datetime.now()

# resistance = np.arange(0.1, 2.5, 0.1)
# compliance = np.arange(0.1, 2.5, 0.1)
# C = 0.8
# DtSimulations = 8.5*3

# triples = []

# for R in resistance:
#     for S in compliance:
#         if (round(R, 2), round(S, 2)) not in [(0.1, 0.1), (0.2, 0.2), (0.2, 0.1),  (0.3, 0.1), (0.4, 0.1), (0.5, 0.1), (0.1, 0.2), (0.1, 0.3)]:
#             triples.append([C, R, S])

# # initialize json file { "C = 0.1" : []}
# with open(f"simulation_results/20240425/results_final_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# with open(f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# print(f"Starting simulations for C = {C}")
# print(datetime.datetime.now())
# # 2024-07-24 15:22:00.569720
# start_time = datetime.datetime.now()
# run_multiple_simulations(triples, DtSimulations, f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", f"simulation_results/20240425/results_final_for_C_{C}.json")
# print(f"Finished simulations for C = {C}")
# print(datetime.datetime.now())
# end_time = datetime.datetime.now()

# resistance = np.arange(0.1, 2.5, 0.1)
# compliance = np.arange(0.1, 2.5, 0.1)
# C = 1.4
# DtSimulations = 8.5*3

# triples = []

# for R in resistance:
#     for S in compliance:
#         if (round(R, 2), round(S, 2)) not in [(0.1, 0.1), (0.2, 0.2), (0.2, 0.1),  (0.3, 0.1), (0.4, 0.1), (0.5, 0.1), (0.1, 0.2), (0.1, 0.3)]:
#             triples.append([C, R, S])

# # initialize json file { "C = 0.1" : []}
# with open(f"simulation_results/20240425/results_final_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# with open(f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# print(f"Starting simulations for C = {C}")
# print(datetime.datetime.now())
# # 2024-07-24 21:58:22.623500
# start_time = datetime.datetime.now()
# run_multiple_simulations(triples, DtSimulations, f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", f"simulation_results/20240425/results_final_for_C_{C}.json")
# print(f"Finished simulations for C = {C}")
# print(datetime.datetime.now())
# end_time = datetime.datetime.now()

# resistance = np.arange(0.1, 2.5, 0.1)
# compliance = np.arange(0.1, 2.5, 0.1)
# C = 0.6
# DtSimulations = 8.5*3

# triples = []

# for R in resistance:
#     for S in compliance:
#         if (round(R, 2), round(S, 2)) not in [(0.1, 0.1), (0.2, 0.2), (0.2, 0.1),  (0.3, 0.1), (0.4, 0.1), (0.5, 0.1), (0.1, 0.2), (0.1, 0.3)]:
#             triples.append([C, R, S])

# # initialize json file { "C = 0.1" : []}
# with open(f"simulation_results/20240425/results_final_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# with open(f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# print(f"Starting simulations for C = {C}")
# print(datetime.datetime.now())
# # 2024-07-25 10:22:51.614151
# start_time = datetime.datetime.now()
# run_multiple_simulations(triples, DtSimulations, f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", f"simulation_results/20240425/results_final_for_C_{C}.json")
# print(f"Finished simulations for C = {C}")
# print(datetime.datetime.now())
# end_time = datetime.datetime.now()

# resistance = np.arange(0.1, 2.5, 0.1)
# compliance = np.arange(0.1, 2.5, 0.1)
# C = 1.6
# DtSimulations = 8.5*3

# triples = []

# for R in resistance:
#     for S in compliance:
#         if (round(R, 2), round(S, 2)) not in [(0.1, 0.1), (0.2, 0.2), (0.2, 0.1),  (0.3, 0.1), (0.4, 0.1), (0.5, 0.1), (0.1, 0.2), (0.1, 0.3)]:
#             triples.append([C, R, S])

# # initialize json file { "C = 0.1" : []}
# with open(f"simulation_results/20240425/results_final_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# with open(f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# print(f"Starting simulations for C = {C}")
# print(datetime.datetime.now())
# # 2024-07-25 16:12:14.764466
# start_time = datetime.datetime.now()
# run_multiple_simulations(triples, DtSimulations, f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", f"simulation_results/20240425/results_final_for_C_{C}.json")
# print(f"Finished simulations for C = {C}")
# print(datetime.datetime.now())
# end_time = datetime.datetime.now()

# resistance = np.arange(0.1, 2.5, 0.1)
# compliance = np.arange(0.1, 2.5, 0.1)
# C = 0.4
# DtSimulations = 8.5*3

# triples = []

# for R in resistance:
#     for S in compliance:
#         if (round(R, 2), round(S, 2)) not in [(0.1, 0.1), (0.2, 0.2), (0.2, 0.1),  (0.3, 0.1), (0.4, 0.1), (0.5, 0.1), (0.1, 0.2), (0.1, 0.3)]:
#             triples.append([C, R, S])

# # initialize json file { "C = 0.1" : []}
# with open(f"simulation_results/20240425/results_final_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# with open(f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# print(f"Starting simulations for C = {C}")
# print(datetime.datetime.now())
# # 2024-07-26 12:52:26.631440
# start_time = datetime.datetime.now()
# run_multiple_simulations(triples, DtSimulations, f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", f"simulation_results/20240425/results_final_for_C_{C}.json")
# print(f"Finished simulations for C = {C}")
# print(datetime.datetime.now())
# end_time = datetime.datetime.now()


# resistance = np.arange(0.1, 2.5, 0.1)
# compliance = np.arange(0.1, 2.5, 0.1)
# C = 1.8
# DtSimulations = 8.5*3

# triples = []

# for R in resistance:
#     for S in compliance:
#         if (round(R, 2), round(S, 2)) not in [(0.1, 0.1), (0.2, 0.2), (0.2, 0.1),  (0.3, 0.1), (0.4, 0.1), (0.5, 0.1), (0.1, 0.2), (0.1, 0.3)]:
#             triples.append([C, R, S])

# # initialize json file { "C = 0.1" : []}
# with open(f"simulation_results/20240425/results_final_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# with open(f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# print(f"Starting simulations for C = {C}")
# print(datetime.datetime.now())
# # 2024-07-26 18:37:09.938516 
# start_time = datetime.datetime.now()
# run_multiple_simulations(triples, DtSimulations, f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", f"simulation_results/20240425/results_final_for_C_{C}.json")
# print(f"Finished simulations for C = {C}")
# print(datetime.datetime.now())
# end_time = datetime.datetime.now()

# resistance = np.arange(0.1, 2.5, 0.1)
# compliance = np.arange(0.1, 2.5, 0.1)
# C = 2
# DtSimulations = 8.5*3

# triples = []

# for R in resistance:
#     for S in compliance:
#         if (round(R, 2), round(S, 2)) not in [(0.1, 0.1), (0.2, 0.2), (0.2, 0.1),  (0.3, 0.1), (0.4, 0.1), (0.5, 0.1), (0.1, 0.2), (0.1, 0.3)]:
#             triples.append([C, R, S])

# # initialize json file { "C = 0.1" : []}
# with open(f"simulation_results/20240425/results_final_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# with open(f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# print(f"Starting simulations for C = {C}")
# print(datetime.datetime.now())
# # 2024-07-26 18:37:09.938516 
# start_time = datetime.datetime.now()
# run_multiple_simulations(triples, DtSimulations, f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", f"simulation_results/20240425/results_final_for_C_{C}.json")
# print(f"Finished simulations for C = {C}")
# print(datetime.datetime.now())
# end_time = datetime.datetime.now()

# resistance = np.arange(0.1, 2.5, 0.1)
# compliance = np.arange(0.1, 2.5, 0.1)
# C = 2.2
# DtSimulations = 8.5*3

# triples = []

# for R in resistance:
#     for S in compliance:
#         if (round(R, 2), round(S, 2)) not in [(0.1, 0.1), (0.2, 0.2), (0.2, 0.1),  (0.3, 0.1), (0.4, 0.1), (0.5, 0.1), (0.1, 0.2), (0.1, 0.3)]:
#             triples.append([C, R, S])

# # initialize json file { "C = 0.1" : []}
# with open(f"simulation_results/20240425/results_final_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# with open(f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# print(f"Starting simulations for C = {C}")
# print(datetime.datetime.now())
# # 2024-07-27 08:35:38.925731
# start_time = datetime.datetime.now()
# run_multiple_simulations(triples, DtSimulations, f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", f"simulation_results/20240425/results_final_for_C_{C}.json")
# print(f"Finished simulations for C = {C}")
# print(datetime.datetime.now())
# end_time = datetime.datetime.now()

# resistance = np.arange(0.1, 2.5, 0.1)
# compliance = np.arange(0.1, 2.5, 0.1)
# C = 2.4
# DtSimulations = 8.5*3

# triples = []

# for R in resistance:
#     for S in compliance:
#         if (round(R, 2), round(S, 2)) not in [(0.1, 0.1), (0.2, 0.2), (0.2, 0.1),  (0.3, 0.1), (0.4, 0.1), (0.5, 0.1), (0.1, 0.2), (0.1, 0.3)]:
#             triples.append([C, R, S])

# # initialize json file { "C = 0.1" : []}
# with open(f"simulation_results/20240425/results_final_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# with open(f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# print(f"Starting simulations for C = {C}")
# print(datetime.datetime.now())
# # 2024-07-27 10:04:12.332738
# start_time = datetime.datetime.now()
# run_multiple_simulations(triples, DtSimulations, f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", f"simulation_results/20240425/results_final_for_C_{C}.json")
# print(f"Finished simulations for C = {C}")
# print(datetime.datetime.now())
# end_time = datetime.datetime.now()


# resistance = np.arange(0.1, 2.5, 0.1)
# compliance = np.arange(0.1, 2.5, 0.1)
# C = 0.2
# DtSimulations = 8.5*3

# triples = []

# for R in resistance:
#     for S in compliance:
#         if (round(R, 2), round(S, 2)) not in [(0.1, 0.1), (0.2, 0.2), (0.2, 0.1),  (0.3, 0.1), (0.4, 0.1), (0.5, 0.1), (0.1, 0.2), (0.1, 0.3)]:
#             triples.append([C, R, S])

# # initialize json file { "C = 0.1" : []}
# with open(f"simulation_results/20240425/results_final_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# with open(f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", 'w') as file:
#     json.dump([], file, indent=4)

# print(f"Starting simulations for C = {C}")
# print(datetime.datetime.now())
# # 2024-07-27 10:04:12.332738
# start_time = datetime.datetime.now()
# run_multiple_simulations(triples, DtSimulations, f"simulation_results/20240425/mid_simulation_results/results_for_C_{C}.json", f"simulation_results/20240425/results_final_for_C_{C}.json")
# print(f"Finished simulations for C = {C}")
# print(datetime.datetime.now())
# end_time = datetime.datetime.now()