import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import json

def read_data():

    list_of_dicts = []
    # stable_part = []

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
            'part': part,
            "MAP" : 1/3 * (np.max(part) - np.min(part)) + np.min(part),
            "flow" : None
        }

        list_of_dicts.append(new_data)

    return list_of_dicts


if __name__ == "__main__":

    data = read_data()
    # df = pd.DataFrame(data)

    