import numpy as np
import pandas as pd

import json


def read_data():
    """
    Iz results.json datoteke preberem podatke tako, da vrnem seznam slovarjev, kjer vsak slovar vsebuje C, R, S in part. 
    To naredimo zato, ker bomo nove podatke parsali na isti način. 
    """

    list_of_lists = []
    stable_part = []

    with open("data/results.json", "r") as json_file:
        data = json.load(json_file)

    # nekatere parametre ful čudno zaokroži
    for key in data:
        params = data[key]['params']
        part = data[key]['part']

        new_data = {
            'C': round(params[0], 3),
            'R': round(params[1], 3),
            'S': round(params[2], 3),
            'part': part
        }

        if new_data['C'] == 1 and new_data['R'] == 1 and new_data['S'] == 1:
            stable_part.append(new_data)
        else:
            list_of_lists.append(new_data)
            
    return list_of_lists, stable_part


def read_new_data():
    """
    Vrne seznam slovarjev, kjer vsak slovar vsebuje C, R, S in part.
    """

    with open("data/results_new.json", "r") as json_file:
        data = json.load(json_file)

    return data

if __name__ == "__main__":
    # read_data()
    data = read_new_data()