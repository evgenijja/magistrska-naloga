import numpy as np
import pandas as pd

import json



def read_new_data():
    """
    Vrne seznam slovarjev, kjer vsak slovar vsebuje C, R, S in part.
    """

    with open("data/results_final.json", "r") as json_file:
        data = json.load(json_file)

    print("Število simulacij:", len(data))

    return data


def filter_data(data):
    """
    Nekatere simulacije so "pokavarjene". Niso v redu delale - vrednosti so prevelike ali pa so samo none. 
    """
    filtered_data = []

    for simulation in data:

        part = simulation['part']
        if not np.isnan(part).any() and np.min(part) >= 0 and len(set(part)) != 1 and np.max(part) < 200:
            filtered_data.append(simulation)

    print("Število simulacij:", len(filtered_data))
    return filtered_data


def read_parsed_data(filter=False):

    with open("data/parsed_data2.json", "r") as json_file:
        data = json.load(json_file)

    print("Število simulacij:", len(data))


    return data


if __name__ == "__main__":
#     # read_data()
    # data = read_new_data()
    # data = filter_data(data)

    data = read_parsed_data()
    data = filter_data(data)

    