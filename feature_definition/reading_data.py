import numpy as np
import pandas as pd

import json

# PARAMS = [c, r, s]

def get_data():
    """
    Dobimo rezultate vseh simulacij. 
    """
    with open("data/results.json", "r") as json_file:
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
    Vrnemo v obliki seznam seznamov - za PCA tko ne rabimo vrednosti paramterov. 
    """

    parts = []
    parts_c, parts_r, parts_s = [], [], []
    for key in data:
        parts.append(data[key]['part'])

        if data[key]['params'][1] == 1 and data[key]['params'][2] == 1:
            parts_c.append(data[key]['part'])

        elif data[key]['params'][0] == 1 and data[key]['params'][2] == 1:
            parts_r.append(data[key]['part']) 

        elif data[key]['params'][1] == 1 and data[key]['params'][0] == 1:
            parts_s.append(data[key]['part']) 
            
    return parts, parts_c, parts_r, parts_s



def normalize_parts(parts):
    """
    Takes a list of lists and normalizes PP to interval 0 to 1
    Returns list of lists
    """
    new_parts = []
    for part in parts:
        pp = np.max(part) - np.min(part)
        normalization_factor = 1 / pp 
        new_part = [part[i] * normalization_factor for i in range(len(part))]
        new_parts.append(new_part)

    return new_parts


def first_translation(parts):
    """
    Move the first part of the curve to the end based on the minimum.
    """
    new_parts = [part[np.argmin(part):] + part[:np.argmin(part)] for part in parts]
    return new_parts

def extract_parts_for_feature_definitions(mode="translated"):
    """
    Vrne seznam slovarjev, vsak slovar vsebuje paramse in pripadajoči val. Različne opcije vrnejo različno sparsane valove:
    - normal: vrne normalne valove
    - translated: vrne valove, ki so bili translirani oziroma je bil prvi del premaknjen na konec
    - translated & normalized: vrne valove, ki so bili translirani in normalizirani na PP = [0, 1]
    - translated residue: vrne razliko med transliranim valom in stabilnim valom
    - translated & normalized residue: vrne razliko med transliranim in normaliziranim valom in stabilnim valom
    - normalized translated residue: najprej odštejemo, nato normaliziramo
    """

    data = get_data()

    new_data = []

    for key in data:
        params = data[key]['params']
        part = data[key]['part']

        if mode == "normal":
            new_data.append({'params': params, 'part': part})

        elif mode == "translated":
            new_data.append({'params': params, 'part': part[np.argmin(part):] + part[:np.argmin(part)]})

        elif mode == "translated & normalized":
            # translate
            part = part[np.argmin(part):] + part[:np.argmin(part)]

            # normalize
            pp = np.max(part) - np.min(part)
            normalization_factor = 1 / pp 
            new_part = [part[i] * normalization_factor for i in range(len(part))]
            new_data.append({'params': params, 'part': new_part})

        elif mode == "translated residue":
            normal_part = get_normal_data(data)
            # translate
            normal_part = normal_part[np.argmin(normal_part):] + normal_part[:np.argmin(normal_part)]
            part = part[np.argmin(part):] + part[:np.argmin(part)]

            new_part = [part[i] - normal_part[i] for i in range(len(part))]
            new_data.append({'params': params, 'part': new_part})

        elif mode == "translated & normalized residue":
            normal_part = get_normal_data(data)
            # translate
            normal_part = normal_part[np.argmin(normal_part):] + normal_part[:np.argmin(normal_part)]
            part = part[np.argmin(part):] + part[:np.argmin(part)]

            # normalize
            pp = np.max(part) - np.min(part)
            normalization_factor = 1 / pp
            new_part = [part[i] * normalization_factor for i in range(len(part))]

            pp_normal = np.max(normal_part) - np.min(normal_part)
            normalization_factor_normal = 1 / pp_normal
            new_normal_part = [normal_part[i] * normalization_factor_normal for i in range(len(normal_part))]
            
            # subtract
            new_part = [new_part[i] - new_normal_part[i] for i in range(len(new_part))]

            new_data.append({'params': params, 'part': new_part})

        elif mode == "normalized translated residue":
            # first subtract, then normalize
            normal_part = get_normal_data(data)
            # translate
            normal_part = normal_part[np.argmin(normal_part):] + normal_part[:np.argmin(normal_part)]
            part = part[np.argmin(part):] + part[:np.argmin(part)]

            # subtract
            new_part = [part[i] - normal_part[i] for i in range(len(part))]

            # normalize
            pp = np.max(new_part) - np.min(new_part)
            normalization_factor = 1 / pp
            new_part = [new_part[i] * normalization_factor for i in range(len(new_part))]
            new_data.append({'params': params, 'part': new_part})

    return new_data


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

# normal_part = get_normal_data(data)
# parts = extract_parts(data)
# normalized_parts = normalize_parts(parts)

# translated_parts = first_translation(parts)
# normalized_translated_parts = normalize_parts(translated_parts)

    

# if __name__=="__main__":

#     data = get_data()

#     # sanity_check(data)

#     normal_part = get_normal_data(data)
#     parts = extract_parts(data)
    