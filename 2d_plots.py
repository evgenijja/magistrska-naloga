import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

def read_parsed_data(filter=False):

    with open("data/parsed_data2.json", "r") as json_file:
        data = json.load(json_file)

    print("Število simulacij:", len(data))

    return data

def filter_out_and_plot(data):
    """
    R v odvisnosti od DBP in SBP za različne pare (C, S)
    """

    cs_pairs = [
        (1.0, 1.0),
        (1.1, 1.1),
        (1.2, 1.2),
        (0.9, 0.9),
        (0.8, 0.8),
        (0.8, 1.2),
        (1.2, 0.8),
        (0.9, 1.1),
        (1.1, 0.9)
    ]

    to_plot = []

    for pair in cs_pairs:
        to_plot_one = []
        for elt in data:
            params = elt['params']
            dbp = np.min(elt['part'])
            sbp = np.max(elt['part'])
            if round(params[0], 2) == pair[0] and round(params[2], 2) == pair[1]:
                to_plot_one.append((round(params[1], 2), dbp, sbp))

        # order tuples by R
        to_plot_one.sort(key=lambda x: x[0])
        to_plot.append(to_plot_one)

    # plot 2 side by side plots where x axis is R, y axis is DBP and SBP
    # on first plot plot all dbp plots in different colors and on the second all sbp plots in different colors

    fig, axs = plt.subplots(1, 2, figsize=(10, 10))

    for i, ax in enumerate(axs):
            
            for j, pair in enumerate(cs_pairs):
                if i == 0:
                    ax.plot([elt[0] for elt in to_plot[j]], [elt[1] for elt in to_plot[j]], label=f"C: {pair[0]}, S: {pair[1]}")
                else:
                    ax.plot([elt[0] for elt in to_plot[j]], [elt[2] for elt in to_plot[j]], label=f"C: {pair[0]}, S: {pair[1]}")
    
            ax.legend()
            ax.set_xlabel("R")
            if i == 0:
                ax.set_ylabel("DBP")
            else:
                ax.set_ylabel("SBP")

    plt.show()


def filter_out_and_plot2(data):
    """
    C v odvisnosti od t_sp 
    """

    rs_pairs = [
        (1.0, 1.0),
        (1.1, 1.1),
        (1.2, 1.2),
        (0.9, 0.9),
        (0.8, 0.8),
        (0.8, 1.2),
        (1.2, 0.8),
        (0.9, 1.1),
        (1.1, 0.9)
    ]

    to_plot = []

    for pair in rs_pairs:
        to_plot_one = []
        for elt in data:
            params = elt['params']
            # index of maximal value in residuum
            t_sp = np.argmax(elt['part']) / len(elt['part'])
            naklon = (np.max(elt['part']) - np.min(elt['part'])) / t_sp
            # dbp = np.min(elt['part'])
            # sbp = np.max(elt['part'])
            if round(params[1], 2) == pair[0] and round(params[2], 2) == pair[1]:
                to_plot_one.append((round(params[0], 2), t_sp, naklon))

        # order tuples by R
        to_plot_one.sort(key=lambda x: x[0])
        to_plot.append(to_plot_one)

    # plot 2 side by side plots where x axis is R, y axis is DBP and SBP
    # on first plot plot all dbp plots in different colors and on the second all sbp plots in different colors

    fig, axs = plt.subplots(1, 2, figsize=(10, 10))

    for i, ax in enumerate(axs):
            
            for j, pair in enumerate(rs_pairs):
                if i == 0:
                    ax.plot([elt[0] for elt in to_plot[j]], [elt[1] for elt in to_plot[j]], label=f"R: {pair[0]}, S: {pair[1]}")
                else:
                    ax.plot([elt[0] for elt in to_plot[j]], [elt[2] for elt in to_plot[j]], label=f"R: {pair[0]}, S: {pair[1]}")
    
            ax.legend()
            ax.set_xlabel("C")
            if i == 0:
                ax.set_ylabel("t_sp")
            else:
                ax.set_ylabel("naklon")

    plt.show()

def filter_out_and_plot3(data):
    """
    S v odvisnosti od sbp, dbp
    """

    rc_pairs = [
        (1.0, 1.0),
        (1.1, 1.1),
        (1.2, 1.2),
        (0.9, 0.9),
        (0.8, 0.8),
        (0.8, 1.2),
        (1.2, 0.8),
        (0.9, 1.1),
        (1.1, 0.9)
    ]

    to_plot = []

    for pair in rc_pairs:
        to_plot_one = []
        for elt in data:
            params = elt['params']
            # index of maximal value in residuum
            # t_sp = np.argmax(elt['part']) / len(elt['part'])
            # naklon = (np.max(elt['part']) - np.min(elt['part'])) / t_sp
            dbp = np.min(elt['part'])
            sbp = np.max(elt['part'])
            if round(params[0], 2) == pair[0] and round(params[1], 2) == pair[1]:
                to_plot_one.append((round(params[2], 2), dbp, sbp))

        # order tuples by R
        to_plot_one.sort(key=lambda x: x[0])
        to_plot.append(to_plot_one)

    # plot 2 side by side plots where x axis is R, y axis is DBP and SBP
    # on first plot plot all dbp plots in different colors and on the second all sbp plots in different colors

    fig, axs = plt.subplots(1, 2, figsize=(10, 10))

    for i, ax in enumerate(axs):
            
            for j, pair in enumerate(rc_pairs):
                if i == 0:
                    ax.plot([elt[0] for elt in to_plot[j]], [elt[1] for elt in to_plot[j]], label=f"C: {pair[0]}, R: {pair[1]}")
                else:
                    ax.plot([elt[0] for elt in to_plot[j]], [elt[2] for elt in to_plot[j]], label=f"C: {pair[0]}, R: {pair[1]}")
    
            ax.legend()
            ax.set_xlabel("S")
            if i == 0:
                ax.set_ylabel("DBP")
            else:
                ax.set_ylabel("sbp")

    plt.show()
    
if __name__=="__main__":
    data = read_parsed_data()
    filter_out_and_plot3(data)



    


