import numpy as np
import pandas as pd

import json

def get_data():
    """
    Dobimo rezultate vseh simulacij. 
    """
    with open("ACTUAL_CODE/data/results.json", "r") as json_file:
        data = json.load(json_file)

    # nekatere parametre ful čudno zaokroži
    for key in data:
        params = data[key]['params']
        new_params = [round(params[0], 3), round(params[1], 3), round(params[2], 3)]
        data[key]['params'] = new_params

    return data

def get_normal_data(data):

    for key in data:
        params = data[key]['params']
        if params == [1, 1, 1]:
            part = data[key]['part']

    return part

def extract_parts(data):
    """
    Funkcija, ki iz slovarja podatkov izloči samo koordinate valov (če ne rabimo label).
    Vrnemo v obliki 
    """
    parts = []
    for key in data:
        parts.append(data[key]['part'])

    return parts


def sanity_check(data):
    """
    Ali imamo parametre samo 1x v množici?
    """

    param_counter = {}
    for key in data:
        params = data[key]['params']
        if tuple(params) in param_counter:
            param_counter[tuple(params)] += 1
        else:
            param_counter[tuple(params)] = 1

    # sorted_dict = dict(sorted(param_counter.items(), key=lambda item: item[1], reverse=True))

    my_dict = param_counter

    key_with_max_value = max(my_dict, key=my_dict.get)

    print("Key with maximum value:", key_with_max_value)
    print("Maximum value:", my_dict[key_with_max_value])

data = get_data()

# sanity_check(data)

normal_part = get_normal_data(data)
parts = extract_parts(data)

    

if __name__=="__main__":

    data = get_data()

    # sanity_check(data)

    normal_part = get_normal_data(data)
    parts = extract_parts(data)