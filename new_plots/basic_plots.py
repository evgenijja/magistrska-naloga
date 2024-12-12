from reading_data import read_parsed_data#, filter_data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline


"""
Želimo narisati:
- 3D plot z območji, kjer je pacient mrtev (prenizek MAP ali prenizek FLOW)

- 2D plote, kjer opazujemo kako se spreminjajo oblike valov glede na spremembe parametrov (perturbations)
    - osi normaliziramo, tako kot Mulder v članku
- gledamo prvi in drugi odvod
- naredimo dekompozicijo valov na ostanke (relativna sprememba glede na stabilen val [1, 1, 1])

"""

def plot_all_simulated_ponts(data):
    """
    Prikažemo vse simulirane točke v prostoru.
    data je seznam slovarjev s ključi params, part in flow
    """

    x, y, z = [], [], []
    x_n, y_n, z_n = [], [], []

    for point in data:
        part = point['part']
        if not np.isnan(part).any() and np.min(part) >= 0 and len(set(part)) != 1:
            x.append(point['params'][0])
            y.append(point['params'][1])
            z.append(point['params'][2])
        else: 
            x_n.append(point['params'][0])
            y_n.append(point['params'][1])
            z_n.append(point['params'][2])
        # x.append(point['params'][0])
        # y.append(point['params'][1])
        # z.append(point['params'][2])

    # for elt in data:
    #     # if elt['part'] doesnt contain any nan values and any negative values and has at least two different values add it to new_data
    #     if not np.isnan(elt['part']).any() and np.min(elt['part']) >= 0 and len(set(elt['part'])) != 1:
    #         new_data.append(elt)
    # print(len(new_data))

    print(len(x))
    print(len(y))
    print(len(z))

    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(x, y, z, c='b')
    sc_n = ax.scatter(x_n, y_n, z_n, c='k')

    ax.set_xlabel('Kontraktilnost')
    ax.set_ylabel('Upornost')
    ax.set_zlabel('Podajnost')

    print(f"Range contractility: {min(x)}, {max(x)}")
    print(f"Range resistance:  {min(y)}, {max(y)}")
    print(f"Range stiffness:  {min(y)}, {max(y)}")
    
    plt.savefig('final_plots/all_pts_new.png')
    plt.show()	


def plot_flow_and_pp(data, flow_threshold=3):
    """
    Vzamemo simulirane točke v prostoru in jih narišemo.
    Barva točk je vrednost MAP.
    """
    # točke, ki bodo imele MAP barvo
    x, y, z = [], [], []

    flow_values, pp_values = [], []

    x_low_flow, y_low_flow, z_low_flow = [], [], []
    low_flow_values, low_pp_values = [], []

    x_pp, y_pp, z_pp = [], [], []
    x_pp_low, y_pp_low, z_pp_low = [], [], []
    low_pp_values = []

    
    for point in data:

        part = point['part']
        map_val = 1/3 * (np.max(part) - np.min(part)) + np.min(part)
        # flow_val =  np.trapz(point['flow'])
        flow_val = np.mean(point['flow'])

        # pretvorba
        flow_val = flow_val * 60000 # m3/s -> l/min

        pp = np.max(part) - np.min(part)

        if np.max(part) - np.min(part) != 0:

            if flow_val < flow_threshold:

                x_low_flow.append(point['params'][0])
                y_low_flow.append(point['params'][1])
                z_low_flow.append(point['params'][2])
                low_flow_values.append(flow_val)
                # low_pp_values.append(pp)

            else:
                x.append(point['params'][0])
                y.append(point['params'][1])
                z.append(point['params'][2])
                flow_values.append(flow_val)
                # pp_values.append(pp)

            if map_val < 65:

                x_pp_low.append(point['params'][0])
                y_pp_low.append(point['params'][1])
                z_pp_low.append(point['params'][2])
                low_pp_values.append(pp)

            else:

                x_pp.append(point['params'][0])
                y_pp.append(point['params'][1])
                z_pp.append(point['params'][2])
                pp_values.append(pp)


    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(x_low_flow, y_low_flow, z_low_flow, color='black', label='PTS')
    

    # make two side by side 3d scatter plots with DIFFERENT colorbars 
    # one for flow and one for pp

    # norm = plt.Normalize(min(flow_values), max(flow_values))
    

    # Plot third set of points with colormap based on map_values
    # sc = ax.scatter(x, y, z, c=flow_values, cmap='viridis', norm=norm, label='PTS')

    # pulse
    # norm = plt.Normalize(min(pp_values), max(pp_values))
    # ax.scatter(x_pp_low, y_pp_low, z_pp_low, color='black', label='PTS')
    # sc = ax.scatter(x_pp, y_pp, z_pp, c=pp_values, cmap='autumn', norm=norm, label='PTS')
    # cbar = plt.colorbar(sc, shrink=0.5)
    # cbar.set_label('pulzni tlak [mmHg]', fontdict={'fontsize': 15})

    # flow
    norm = plt.Normalize(min(flow_values), max(flow_values))
    ax.scatter(x_low_flow, y_low_flow, z_low_flow, color='black', label='PTS')
    sc = ax.scatter(x, y, z, c=flow_values, cmap='viridis', norm=norm, label='PTS')
    cbar = plt.colorbar(sc, shrink=0.5)
    cbar.set_label('pretok [l/min]', fontdict={'fontsize': 15})


    # Add a colorbar for the third set
    
    
    
    # change size of the colorbar
    cbar.ax.tick_params(labelsize=15)


    # Labels
    ax.set_xlabel('Kontraktilnost', fontdict={'fontsize': 15})
    ax.set_ylabel('Upornost', fontdict={'fontsize': 15})

    ax.zaxis.set_rotate_label(False) 
    ax.set_zlabel('Podajnost', fontdict={'fontsize': 15, 'rotation': 90})

    # set fixed size of axes
    ax.set_xlim(0.55, 2.15)
    ax.set_ylim(0.7, 1.5)
    ax.set_zlim(0.1, 1.3)

    # tilte
    plt.title('PRETOK', fontdict={'fontsize': 20})

    # plt.colorbar(im,)

    plt.savefig('final_plots/flow_pts.png')
    plt.tight_layout()
    plt.show()


def plot_death_points(data, map_threshold=65, flow_threshold=0.00005, map_upper_threshold=150):
    """
    Vzamemo simulirane točke v prostoru in jih narišemo.
    Barva točk je vrednost MAP.
    """
    # točke, ki bodo imele MAP barvo
    x, y, z = [], [], []
    map_values, flow_values = [], []

    # točke, ki bodo črne (prenizek map)
    x_map, y_map, z_map = [], [], []
    map_values_map, flow_values_map = [], []

    # točke, ki bodo rjave (prenizek flow)
    x_flow, y_flow, z_flow = [], [], []
    map_values_flow, flow_values_flow = [], []

    # točke, ki bodo svetlo sive (previsok map)
    x_map_upper, y_map_upper, z_map_upper = [], [], []
    map_values_map_upper, flow_values_map_upper = [], []
    
    for point in data:

        part = point['part']
        map_val = 1/3 * (np.max(part) - np.min(part)) + np.min(part)
        flow_val =  np.trapz(point['flow'])

        if map_val < map_threshold:
            x_map.append(point['params'][0])
            y_map.append(point['params'][1])
            z_map.append(point['params'][2])
            map_values_map.append(map_val)
            flow_values_map.append(flow_val)

        elif flow_val < flow_threshold:
            x_flow.append(point['params'][0])
            y_flow.append(point['params'][1])
            z_flow.append(point['params'][2])
            map_values_flow.append(map_val)
            flow_values_flow.append(flow_val)

        elif map_val > map_upper_threshold:
            x_map_upper.append(point['params'][0])
            y_map_upper.append(point['params'][1])
            z_map_upper.append(point['params'][2])
            map_values_map_upper.append(map_val)
            flow_values_map_upper.append(flow_val)

        else:
            x.append(point['params'][0])
            y.append(point['params'][1])
            z.append(point['params'][2])
            map_values.append(map_val)
            flow_values.append(flow_val)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    print(len(x_map))
    print(len(x_flow))
    print(len(x_map_upper))

    # print(x_map_upper)
    # print(y_map_upper)
    # print(z_map_upper)

    print(f"min value of map_values_map: {max(map_values_map_upper)}")

    # print ordered map_values_map from lowest to highest
    print(sorted(map_values_map))
    # print(map_values_map)



    # Plot first set of points in black
    ax.scatter(x_map, y_map, z_map, color='black', label=f'povprečen krvni tlak < {map_threshold} mmHg')

    # Plot second set of points in brown
    ax.scatter(x_flow, y_flow, z_flow, color='black', label=f'pretok < {flow_threshold} m3/s')

    # Plot third set of points in light grey
    ax.scatter(x_map_upper, y_map_upper, z_map_upper, color='lightgrey', label=f'povprečen krvni tlak > {map_upper_threshold} mmHg')



    norm = plt.Normalize(min(map_values), max(map_values))

    # Plot third set of points with colormap based on map_values
    sc = ax.scatter(x, y, z, c=map_values, cmap='autumn', norm=norm, label='PTS')

    # Add a colorbar for the third set
    cbar = plt.colorbar(sc, shrink=0.5)
    cbar.set_label('srednji krvni tlak [mmHg]', fontdict={'fontsize': 15})

    # Labels
    ax.set_xlabel('Kontraktilnost', fontdict={'fontsize': 15})
    ax.set_ylabel('Upornost', fontdict={'fontsize': 15})
    ax.set_zlabel('Podajnost', fontdict={'fontsize': 15})



    # ranges: 
    # C : 0.55 - 2.15
    # R : 0.7 - 1.5
    # S :  0.1 - 1.3

    # set fixed size of axes
    ax.set_xlim(0.55, 2.15)
    ax.set_ylim(0.7, 1.5)
    ax.set_zlim(0.1, 1.3)

    plt.savefig('final_plots/death_pts_test.png')

    plt.show()

def plot_death_points_side_by_side(data, map_threshold=65, flow_threshold=0.00005, map_upper_threshold=150):
    """
    Vzamemo simulirane točke v prostoru in jih narišemo.
    Barva točk je vrednost MAP.
    """
    # točke, ki bodo imele MAP barvo
    x, y, z = [], [], []
    map_values, flow_values = [], []

    # točke, ki bodo črne (prenizek map)
    x_map, y_map, z_map = [], [], []
    map_values_map, flow_values_map = [], []

    # točke, ki bodo rjave (prenizek flow)
    x_flow, y_flow, z_flow = [], [], []
    map_values_flow, flow_values_flow = [], []

    # točke, ki bodo svetlo sive (previsok map)
    x_map_upper, y_map_upper, z_map_upper = [], [], []
    map_values_map_upper, flow_values_map_upper = [], []
    
    for point in data:

        part = point['part']
        map_val = 1/3 * (np.max(part) - np.min(part)) + np.min(part)
        flow_val =  np.trapz(point['flow'])

        if map_val < map_threshold:
            x_map.append(point['params'][0])
            y_map.append(point['params'][1])
            z_map.append(point['params'][2])
            map_values_map.append(map_val)
            flow_values_map.append(flow_val)

        elif flow_val < flow_threshold:
            x_flow.append(point['params'][0])
            y_flow.append(point['params'][1])
            z_flow.append(point['params'][2])
            map_values_flow.append(map_val)
            flow_values_flow.append(flow_val)

        elif map_val > map_upper_threshold:
            x_map_upper.append(point['params'][0])
            y_map_upper.append(point['params'][1])
            z_map_upper.append(point['params'][2])
            map_values_map_upper.append(map_val)
            flow_values_map_upper.append(flow_val)

        else:
            x.append(point['params'][0])
            y.append(point['params'][1])
            z.append(point['params'][2])
            map_values.append(map_val)
            flow_values.append(flow_val)

    # create two side by side 3d scatter plots
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # Flatten the 2x3 grid of axes for easy iteration
    axs = axs.flatten()

    # Plot first set of points in black
    axs[0].scatter(x_map, y_map, z_map, color='black', label=f'povprečen krvni tlak < {map_threshold} mmHg')

    # Plot second set of points in brown
    axs[0].scatter(x_flow, y_flow, z_flow, color='black', label=f'pretok < {flow_threshold} m3/s')

    # Plot third set of points in light grey
    axs[0].scatter(x_map_upper, y_map_upper, z_map_upper, color='lightgrey', label=f'povprečen krvni tlak > {map_upper_threshold} mmHg')

    norm = plt.Normalize(min(map_values), max(map_values))

    # Plot third set of points with colormap based on map_values
    sc = axs[0].scatter(x, y, z, c=map_values, cmap='autumn', norm=norm, label='PTS')
    
    # Add a colorbar for the third set
    cbar = plt.colorbar(sc, ax=axs[0])
    cbar.set_label('Vrednost srednjega krvnega tlaka', fontdict={'fontsize': 15})

    # Labels
    axs[0].set_xlabel('Kontraktilnost', fontdict={'fontsize': 15})
    axs[0].set_ylabel('Upornost', fontdict={'fontsize': 15})
    # axs[0].zaxis.set_rotate_label(False) 
    # axs[0].set_zlabel('Podajnost', fontdict={'fontsize': 15})

    # set fixed size of axes
    axs[0].set_xlim(0.55, 2.15)
    axs[0].set_ylim(0.7, 1.5)
    axs[0].set_zlim(0.1, 1.3)

    # Plot first set of points in black
    axs[1].scatter(x_map, y_map, z_map, color='black', label=f'povprečen krvni tlak < {map_threshold} mmHg')

    # Plot second set of points in brown
    axs[1].scatter(x_flow, y_flow, z_flow, color='black', label=f'pretok < {flow_threshold} m3/s')

    # Plot third set of points in light grey
    axs[1].scatter(x_map_upper, y_map_upper, z_map_upper, color='lightgrey', label=f'povprečen krvni tlak > {map_upper_threshold} mmHg')

    norm = plt.Normalize(min(flow_values), max(flow_values))

    # Plot third set of points with colormap based on map_values

    sc = axs[1].scatter(x, y, z,  c=map_values, cmap='autumn', norm=norm, label='PTS')

    # Labels
    axs[1].set_xlabel('Kontraktilnost', fontdict={'fontsize': 15})
    axs[1].set_ylabel('Upornost', fontdict={'fontsize': 15})
    # axs[1].zaxis.set_rotate_label(False) 
    # axs[1].set_zlabel('Podajnost', fontdict={'fontsize': 15})

    # set fixed size of axes
    axs[1].set_xlim(0.55, 2.15)
    axs[1].set_ylim(0.7, 1.5)
    axs[1].set_zlim(0.1, 1.3)




def plot_just_one_wave(data):
    """
    Narišemo samo en val, ki ga izberemo iz data.
    """
    for point in data:
        params = point['params']
        params = [round(param, 2) for param in params]
        if params == [1, 1, 1]:
            part = point['part']

    # plot 3 plots side by side
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Flatten the 2x3 grid of axes for easy iteration
    axs = axs.flatten()

    # add title for each subplot
    axs[0].set_title('Simuliran val', fontsize=18)
    axs[1].set_title('Transformacija vala', fontsize=18)
    axs[2].set_title('Normalizacija \n transformiranega vala', fontsize=18)

    # set names for x and y axis
    axs[0].set_xlabel('čas [s]', fontsize=15)
    axs[0].set_ylabel('arterijski pulzni tlak [mmHg]', fontsize=15)
    axs[1].set_xlabel('čas [s]', fontsize=15)
    axs[1].set_ylabel('arterijski pulzni tlak [mmHg]', fontsize=15)
    axs[2].set_xlabel('normaliziran čas', fontsize=15)
    axs[2].set_ylabel('normaliziran arterijski\n pulzni tlak', fontsize=15)
    # axs[2].set_ylabel('ABP [mmHg]')

    # on the first one just plot the part, x axis should be of lenth the same as part but go from 0 to 0.85
    len_part = len(part)
    x_axis = np.linspace(0, 0.85, len_part)
    axs[0].plot(x_axis, part)

    # on the second one plot the transformed_part
    part_transformed = part[np.argmin(part):] + part[:np.argmin(part)]
    axs[1].plot(x_axis, part_transformed)

    # on the third one plot the normalized part
    part_normalized = [(part_transformed[i]-np.min(part_transformed)) / (np.max(part_transformed)-np.min(part_transformed)) for i in range(len_part)]
    x_axis = np.linspace(0, 1, len_part)
    axs[2].plot(x_axis, part_normalized)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    plt.savefig('final_plots/just_one_wave.png')

    # Display the plot
    plt.show()


def plot_single_wave_for_features(data):

    for point in data:
        params = point['params']
        params = [round(param, 2) for param in params]
        if params == [1, 1, 1]:
            part = point['part']
            part_transformed = part[np.argmin(part):] + part[:np.argmin(part)]

    # x axis should be of length the same as part but go from 0 to 0.85
    len_part = len(part)
    x_axis = np.linspace(0, 0.85, len_part)

    map_val = 1/3 * (np.max(part) - np.min(part)) + np.min(part)

    # plot horizontal line at map_val
    plt.axhline(y=map_val, color='orange', linestyle='--')

    # plot horizontal line at min and max
    plt.axhline(y=np.min(part), color='green', linestyle='--')
    plt.axhline(y=np.max(part), color='red', linestyle='--')

    # plot a vertical line at the index of the maximum of the transformed part that only goes from min to max
    plt.axvline(x=x_axis[np.argmax(part_transformed)], ymin=np.min(part_transformed), ymax=np.max(part_transformed), color='blue', linestyle='--')


    plt.plot(x_axis, part_transformed, c='black')
    # axis labels
    plt.xlabel('čas [s]', fontsize=15)
    plt.ylabel('ABP [mmHg]', fontsize=15)
    plt.show()


def plot_2d(data, normalize_x_axis=True, normalize_y_axis=False):
    """
    2d graf kjer sta dve spremenljivki fiksni, ena pa se spreminja (3 grafi), en navaden graf in en normaliziran na y os

    Za vsak graf še perturbacija vsake druge spremenljivke za 1+2 in 3+4 korake?

    Za vsak graf še perturabcija obeh spremenljivk

    Naj bodo možnost da je x os od 0 do 1
    """

    if normalize_x_axis:
        len_part = len(data[0]['part'])

        # dolžina enega srčnega utripa 0.85 sekunde?
        x_axis = np.linspace(0, 0.85, len_part)

    fixed_contractility, fixed_resistance, fixed_stiffness = [], [], []
    fixed_contractility_transformed, fixed_resistance_transformed, fixed_stiffness_transformed = [], [], []
    fixed_contractility_normalized, fixed_resistance_normalized, fixed_stiffness_normalized = [], [], []

    stable_part, stable_part_transformed, stable_part_normalized = [], [], []
    
    for point in data:
        params = point['params']
        part_transformed = point['part'][np.argmin(point['part']):] + point['part'][:np.argmin(point['part'])]
        map_val = 1/3 * (np.max(point['part']) - np.min(point['part'])) + np.min(point['part'])

        if round(params[0], 2) == 1 and round(params[1], 2) == 1 and round(params[2], 2) == 1:
            stable_part.append(point['part'])
            stable_part_transformed.append(part_transformed)
            part_transformed = [part_transformed[i] - np.min(part_transformed) for i in range(len_part)]
            stable_part_normalized.append([part_transformed[i] / np.max(part_transformed) for i in range(len_part)])

        # if round(params[0], 2) == 1 and round(params[1], 2) == 1 and round(params[2], 2) != 1:
        #     fixed_contractility.append(point['part'])
        #     fixed_contractility_transformed.append(part_transformed)
        #     fixed_contractility_normalized.append([part_transformed[i] / np.max(part_transformed) for i in range(len_part)])
        if round(params[0], 2) == 1 and round(params[2], 2) == 1 and round(params[1], 2) != 1:
            fixed_resistance.append(point['part'])
            fixed_resistance_transformed.append(part_transformed)
            fixed_resistance_normalized.append([part_transformed[i] / np.max(part_transformed) for i in range(len_part)])
        elif round(params[1], 2) == 1 and round(params[2], 2) == 1 and round(params[0], 2) != 1:
            fixed_contractility.append(point['part'])
            fixed_contractility_transformed.append(part_transformed)
            fixed_contractility_normalized.append([part_transformed[i] / np.max(part_transformed) for i in range(len_part)])
        elif round(params[0], 2) == 1 and round(params[1], 2) == 1 and round(params[2], 2) != 1:
            fixed_stiffness.append(point['part'])
            fixed_stiffness_transformed.append(part_transformed)
            fixed_stiffness_normalized.append([part_transformed[i] / np.max(part_transformed) for i in range(len_part)])
        

        # if round(params[0], 2) == 1 and round(params[1], 2) == 1 and round(params[2], 2) == 1:
        #     print("found it")
        #     fixed_resistance.append(point['part'])
        #     fixed_resistance_transformed.append(part_transformed)
        #     fixed_resistance_normalized.append([part_transformed[i] / np.max(part_transformed) for i in range(len_part)])

        #     fixed_stiffness.append(point['part'])
        #     fixed_stiffness_transformed.append(part_transformed)
        #     fixed_stiffness_normalized.append([part_transformed[i] / np.max(part_transformed) for i in range(len_part)])

    plots = [
        # fixed_contractility, 
        # fixed_resistance, 
        # fixed_stiffness, 
        fixed_contractility, 
        fixed_resistance, 
        fixed_stiffness, 
        fixed_contractility, 
        fixed_resistance, 
        fixed_stiffness, 
        # fixed_contractility_transformed,
        # fixed_resistance_transformed,
        # fixed_stiffness_transformed,
        # fixed_contractility_normalized, 
        # fixed_resistance_normalized, 
        # fixed_stiffness_normalized,
    ]

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    # Flatten the 2x3 grid of axes for easy iteration
    axs = axs.flatten()
    cmap = plt.get_cmap('autumn')

    # Loop through the data and corresponding subplot axis
    for num, parts in enumerate(plots):
        # parts is a list of parts
        # to bo v resnici delovalo samo za prve tri ker računa direkt
        map_values = [1/3 * (np.max(part) - np.min(part)) + np.min(part) for part in parts]
        # pts = np.linspace(min(map_values), max(map_values), len(map_values))

        # generate a list of all rounded values between min(map_values) and max(map_values)
        map_list = [val for val in range(round(min(map_values)), round(max(map_values)+1))]
        ordered_list = np.linspace(0, 1, len(map_list))

        color_dict = dict((map_list[i], ordered_list[i]) for i in range(len(map_list)))
        colors = plt.cm.YlOrRd(ordered_list)

        # if num in [0,1,2]:
        #     new_parts = parts

        if num in [0, 1, 2]:
            new_parts = [part[np.argmin(part):] + part[:np.argmin(part)] for part in parts]

        if num in [3, 4, 5]:
            new_parts = [part[np.argmin(part):] + part[:np.argmin(part)] for part in parts]
            # new_parts = [[(new_parts[i][j] - min(new_parts[i])) / max(new_parts[i]) for j in range(len(new_parts[i]))] for i in range(len(new_parts))]
            new_parts = [[(new_parts[i][j]-np.min(new_parts[i])) /(np.max(new_parts[i])-np.min(new_parts[i])) for j in range(len(new_parts[i]))] for i in range(len(new_parts))]
        
        if num in [2, 5]:
            x_axis = np.linspace(0, 1, len_part)
        else:
            x_axis = np.linspace(0, 0.85, len_part)

        for i in range(len(parts)):
            part = parts[i]
            map_val = round(1/3 * (np.max(part) - np.min(part)) + np.min(part))
            new_part = new_parts[i]
            ind = np.where(ordered_list == color_dict[map_val])
            axs[num].plot(x_axis, new_part, color=colors[ind])
            # axs[num].plot(x_axis, new_part)

    x_axis2 = np.linspace(0, 1, len_part)
    x_axis = np.linspace(0, 0.85, len_part)

    # dashed black line for stable part
    axs[0].plot(x_axis, stable_part_transformed[0], color='black', linestyle='dashed')
    axs[1].plot(x_axis, stable_part_transformed[0], color='black', linestyle='dashed')
    axs[2].plot(x_axis2, stable_part_transformed[0], color='black', linestyle='dashed')

    axs[3].plot(x_axis, stable_part_normalized[0], color='black', linestyle='dashed')
    axs[4].plot(x_axis, stable_part_normalized[0], color='black', linestyle='dashed')
    axs[5].plot(x_axis2, stable_part_normalized[0], color='black', linestyle='dashed')



        # axs[num].set_title(f'Plot {i+1}')

    # add title for each subplot
    axs[0].set_title('Spreminjamo\n kontraktilnost', fontsize=18)
    axs[1].set_title('Spreminjamo\n upornost', fontsize=18)
    axs[2].set_title('Spreminjamo\n podajnost', fontsize=18)

    # set names for x and y axis
    axs[0].set_xlabel('čas [s]', fontsize=18)
    axs[0].set_ylabel('ABP [mmHg]', fontsize=18)
    axs[1].set_xlabel('čas [s]', fontsize=18)
    axs[1].set_ylabel('ABP [mmHg]', fontsize=18)
    axs[2].set_xlabel('čas [s]', fontsize=18)
    axs[2].set_ylabel('ABP [mmHg]', fontsize=18)

    axs[3].set_ylabel('ABP [mmHg]', fontsize=18)
    axs[4].set_ylabel('ABP [mmHg]',     fontsize=18)
    axs[5].set_ylabel('normaliziran ABP', fontsize=18)

    axs[3].set_xlabel('čas [s]', fontsize=18)
    axs[4].set_xlabel('čas [s]', fontsize=18)
    axs[5].set_xlabel('normaliziran čas',   fontsize=18)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the plot
    plt.show()

def plot_2d_res(data, normalize_x_axis=True, normalize_y_axis=False):
    """
    2d graf kjer sta dve spremenljivki fiksni, ena pa se spreminja (3 grafi), en navaden graf in en normaliziran na y os

    Za vsak graf še perturbacija vsake druge spremenljivke za 1+2 in 3+4 korake?

    Za vsak graf še perturabcija obeh spremenljivk

    Naj bodo možnost da je x os od 0 do 1
    """

    if normalize_x_axis:
        len_part = len(data[0]['part'])

        # dolžina enega srčnega utripa 0.85 sekunde?
        x_axis = np.linspace(0, 0.85, len_part)

    fixed_contractility, fixed_resistance, fixed_stiffness = [], [], []
    fixed_contractility_transformed, fixed_resistance_transformed, fixed_stiffness_transformed = [], [], []
    fixed_contractility_normalized, fixed_resistance_normalized, fixed_stiffness_normalized = [], [], []

    fixed_contractility_res, fixed_resistance_res, fixed_stiffness_res = [], [], []
    fixed_contractility_res_normalized, fixed_resistance_res_normalized, fixed_stiffness_res_normalized = [], [], []

    # stable_part, stable_part_transformed, stable_part_normalized = [], [], []

    for point in data:

        params = point['params']
        if round(params[0], 2) == 1 and round(params[1], 2) == 1 and round(params[2], 2) == 1:
            stable_part = point['part']


            first_part, second_part = point['part'][np.argmin(point['part']):], point['part'][:np.argmin(point['part'])]
            print((len(first_part), len(second_part)))
            if len(first_part) != 0 and len(second_part) != 0:
                diff = second_part[-1] - first_part[0]
                stable_part_transformed = second_part + [val + diff for val in first_part]


            # part_transformed = [stable_part_transformed[i] - np.min(stable_part_transformed) for i in range(len_part)]
            # stable_part_normalized = [part_transformed[i] / np.max(part_transformed) for i in range(len_part)]
            stable_part_map = 1/3 * (np.max(point['part']) - np.min(point['part'])) + np.min(point['part'])

    problematic = []
    for point in data:
        params = point['params']

        # transform part continously
        first_part, second_part = point['part'][np.argmin(point['part']):], point['part'][:np.argmin(point['part'])]
        # print((len(first_part), len(second_part)))
        if len(first_part) != 0 and len(second_part) != 0:
            diff = second_part[-1] - first_part[0]
            part_transformed = second_part + [val + diff for val in first_part]
        else:
            problematic.append(point)
            part_transformed = point['part']
            # continue
        

        # part_transformed = point['part'][np.argmin(point['part']):] + point['part'][:np.argmin(point['part'])]
        map_val = 1/3 * (np.max(point['part']) - np.min(point['part'])) + np.min(point['part'])

        # if round(params[0], 2) == 1 and round(params[1], 2) == 1 and round(params[2], 2) == 1:
        #     stable_part.append(point['part'])
        #     stable_part_transformed.append(part_transformed)
        #     part_transformed = [part_transformed[i] - np.min(part_transformed) for i in range(len_part)]
        #     stable_part_normalized.append([part_transformed[i] / np.max(part_transformed) for i in range(len_part)])

        # if round(params[0], 2) == 1 and round(params[1], 2) == 1 and round(params[2], 2) != 1:
        #     fixed_contractility.append(point['part'])
        #     fixed_contractility_transformed.append(part_transformed)
        #     fixed_contractility_normalized.append([part_transformed[i] / np.max(part_transformed) for i in range(len_part)])
        if round(params[0], 2) == 1 and round(params[2], 2) == 1 and round(params[1], 2) != 1:
            fixed_resistance.append(point['part'])
            fixed_resistance_transformed.append(part_transformed)
            fixed_resistance_normalized.append([part_transformed[i] / np.max(part_transformed) for i in range(len_part)])

            part = [part_transformed[i] - map_val for i in range(len_part)]
            stable_part_minus_map = [stable_part_transformed[i] - stable_part_map for i in range(len_part)]

            res = [part[i] - stable_part_minus_map[i] for i in range(len_part)]
            normalized_res = [(res[i]-np.min(res)) / (np.max(res)-np.min(res)) for i in range(len_part)]

            fixed_resistance_res.append(res)
            fixed_resistance_res_normalized.append(normalized_res)

        elif round(params[1], 2) == 1 and round(params[2], 2) == 1 and round(params[0], 2) != 1:
            fixed_contractility.append(point['part'])
            fixed_contractility_transformed.append(part_transformed)
            fixed_contractility_normalized.append([part_transformed[i] / np.max(part_transformed) for i in range(len_part)])

            part = point['part']
            res = [part[i] - stable_part[i] for i in range(len_part)]
            normalized_res = [(res[i]-np.min(res)) / (np.max(res)-np.min(res)) for i in range(len_part)]
            fixed_contractility_res.append(res)
            fixed_contractility_res_normalized.append(normalized_res)

        elif round(params[0], 2) == 1 and round(params[1], 2) == 1 and round(params[2], 2) != 1:
            fixed_stiffness.append(point['part'])
            fixed_stiffness_transformed.append(part_transformed)
            fixed_stiffness_normalized.append([part_transformed[i] / np.max(part_transformed) for i in range(len_part)])

            part = point['part']
            res = [part[i] - stable_part[i] for i in range(len_part)]
            normalized_res = [(res[i]-np.min(res)) / (np.max(res)-np.min(res)) for i in range(len_part)]
            fixed_stiffness_res.append(res)
            fixed_stiffness_res_normalized.append(normalized_res)
        

        # if round(params[0], 2) == 1 and round(params[1], 2) == 1 and round(params[2], 2) == 1:
        #     print("found it")
        #     fixed_resistance.append(point['part'])
        #     fixed_resistance_transformed.append(part_transformed)
        #     fixed_resistance_normalized.append([part_transformed[i] / np.max(part_transformed) for i in range(len_part)])

        #     fixed_stiffness.append(point['part'])
        #     fixed_stiffness_transformed.append(part_transformed)
        #     fixed_stiffness_normalized.append([part_transformed[i] / np.max(part_transformed) for i in range(len_part)])

    plots = [
        # fixed_contractility, 
        # fixed_resistance, 
        # fixed_stiffness, 
        fixed_contractility_transformed, 
        fixed_resistance_transformed, 
        fixed_stiffness_transformed, 
        fixed_contractility_res,
        fixed_resistance_res,
        fixed_stiffness_res,
        fixed_contractility_res_normalized,
        fixed_resistance_res_normalized,
        fixed_stiffness_res_normalized,
        # fixed_contractility, 
        # fixed_resistance, 
        # fixed_stiffness, 
        # fixed_contractility_transformed,
        # fixed_resistance_transformed,
        # fixed_stiffness_transformed,
        # fixed_contractility_normalized, 
        # fixed_resistance_normalized, 
        # fixed_stiffness_normalized,
    ]

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))

    # Flatten the 2x3 grid of axes for easy iteration
    axs = axs.flatten()
    cmap = plt.get_cmap('autumn')

    # Loop through the data and corresponding subplot axis
    for num, parts in enumerate(plots):
        # parts is a list of parts
        # to bo v resnici delovalo samo za prve tri ker računa direkt
        map_values = [1/3 * (np.max(part) - np.min(part)) + np.min(part) for part in parts]
        # pts = np.linspace(min(map_values), max(map_values), len(map_values))

        # generate a list of all rounded values between min(map_values) and max(map_values)
        map_list = [val for val in range(round(min(map_values)), round(max(map_values)+1))]
        ordered_list = np.linspace(0, 1, len(map_list))

        color_dict = dict((map_list[i], ordered_list[i]) for i in range(len(map_list)))
        colors = plt.cm.YlOrRd(ordered_list)

        # if num in [0,1,2]:
        #     new_parts = parts

        # if num in [0, 1, 2]:
        #     new_parts = [part[np.argmin(part):] + part[:np.argmin(part)] for part in parts]

        # if num in [3, 4, 5]:
        #     new_parts = [part[np.argmin(part):] + part[:np.argmin(part)] for part in parts]
        #     # new_parts = [[(new_parts[i][j] - min(new_parts[i])) / max(new_parts[i]) for j in range(len(new_parts[i]))] for i in range(len(new_parts))]
        #     new_parts = [[(new_parts[i][j]-np.min(new_parts[i])) /(np.max(new_parts[i])-np.min(new_parts[i])) for j in range(len(new_parts[i]))] for i in range(len(new_parts))]
        
        new_parts = parts

        for i in range(len(parts)):
            if i in [2, 5]:
                x_axis2 = np.linspace(0, 1, len_part)
                part = parts[i]
                map_val = round(1/3 * (np.max(part) - np.min(part)) + np.min(part))
                new_part = new_parts[i]
                ind = np.where(ordered_list == color_dict[map_val])
                axs[num].plot(x_axis2, new_part, color=colors[ind])
            else:
                part = parts[i]
                map_val = round(1/3 * (np.max(part) - np.min(part)) + np.min(part))
                new_part = new_parts[i]
                ind = np.where(ordered_list == color_dict[map_val])
                axs[num].plot(x_axis, new_part, color=colors[ind])
                # axs[num].plot(x_axis, new_part)

    x_axis2 = np.linspace(0, 1, len_part)

    # dashed black line for stable part
    axs[0].plot(x_axis, stable_part_transformed, color='black', linestyle='dashed')
    axs[1].plot(x_axis, stable_part_transformed, color='black', linestyle='dashed')
    axs[2].plot(x_axis2, stable_part_transformed, color='black', linestyle='dashed')

    const = [0] * len(stable_part_transformed)
    axs[3].plot(x_axis, const, color='black', linestyle='dashed')
    axs[4].plot(x_axis, const, color='black', linestyle='dashed')
    axs[5].plot(x_axis2, const, color='black', linestyle='dashed')



        # axs[num].set_title(f'Plot {i+1}')

    # add title for each subplot
    axs[0].set_title('Spreminjamo kontraktilnost', fontsize=18)
    axs[1].set_title('Spreminjamo upornost', fontsize=18)
    axs[2].set_title('Spreminjamo podajnost', fontsize=18)

    # set names for x and y axis
    axs[0].set_xlabel('čas [s]', fontsize=18)
    axs[0].set_ylabel('arterijski pulzni tlak [mmHg]', fontsize=15)
    axs[1].set_xlabel('čas [s]', fontsize=18)
    axs[1].set_ylabel('arterijski pulzni tlak [mmHg]', fontsize=15)
    axs[2].set_xlabel('normaliziran čas', fontsize=18)
    axs[2].set_ylabel('normaliziran arterijski\n pulzni tlak', fontsize=15)


    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the plot
    plt.show()

def plot_main_points(data):
    for point in data:

        params = point['params']
        if round(params[0], 2) == 1 and round(params[1], 2) == 1 and round(params[2], 2) == 1:
            stable_part = point['part']
            first_part, second_part = point['part'][np.argmin(point['part']):], point['part'][:np.argmin(point['part'])]
            stable_part_transformed = stable_part[np.argmin(stable_part):] + stable_part[:np.argmin(stable_part)]

            # index of maximum
            systolic_peak_index = np.argmax(stable_part_transformed)

            # index of maximum 2nd derivative
            x = np.arange(len(first_part))
            # print(first_part))
            spl = CubicSpline(x, first_part)
            y_new = spl(x)
            deriv1 = spl(x, nu=1)
            deriv2 = spl(x, nu=2)

            dicrotic_notch_index = np.argmax(deriv2[systolic_peak_index:]) + systolic_peak_index

            x = np.arange(len(stable_part_transformed))
            spl = CubicSpline(x, stable_part_transformed)
            y_new = spl(x)
            deriv1 = spl(x, nu=1)
            deriv2 = spl(x, nu=2)


    # plot line plot stable_part and then scatter plot systolic_peak_index and dicrotic_notch_index with different shapes!
    plt.plot(stable_part_transformed, 'k')
    # plt.plot(deriv1, 'r')
    # plt.plot(deriv2, 'b')



    # change size od markers
    plt.scatter(systolic_peak_index, stable_part_transformed[systolic_peak_index], marker='o', s=100, label='sistolni vrh')
    plt.scatter(dicrotic_notch_index+3, stable_part_transformed[dicrotic_notch_index+3], marker='o', color='orange', s=100, label='dikrotska zareza')
    # draw vertical dashed line at systolic peak index that goes from min(part) to max(part) wit arrows on each end
    # plt.vlines(systolic_peak_index, np.min(stable_part_transformed), np.max(stable_part_transformed), linestyles='dashed', colors='green', label='pulzni tlak')
    # plt.hlines(stable_part_transformed[systolic_peak_index], 0, len(stable_part_transformed), linestyles='dashed')
    plt.legend()
    plt.show()

    # fig, ax1 = plt.subplots()

    # # Plot the first dataset
    # ax1.plot(x, stable_part_transformed, 'b-', label='Y1')
    # ax1.set_xlabel('X-axis')
    # ax1.set_ylabel('Y1', color='b')

    # # Create a second y-axis sharing the same x-axis
    # ax2 = ax1.twinx()
    # ax2.plot(x, deriv1, 'r-', label='Y2')
    # ax2.set_ylabel('Y2', color='r')

    # # Create a third y-axis sharing the same x-axis
    # ax3 = ax1.twinx()
    # ax3.spines['right'].set_position(('outward', 60))  # Offset the third y-axis
    # ax3.plot(x, deriv2, 'g-', label='Y3')
    # ax3.set_ylabel('Y3', color='g')

    # # Add a legend for clarity
    # fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

    # plt.show()



def plot_derviatives():
    pass 


def plot_decomposition(data, color_dict):
    """
    Residuum
    """

    for elt in data:
        if [round(elt['params'][0], 2), round(elt['params'][1], 2), round(elt['params'][2], 2)] == [1, 1, 1]:
            stable_part = elt['part']

            # transform 
            first_part, second_part = elt['part'][np.argmin(elt['part']):], elt['part'][:np.argmin(elt['part'])]
            diff = second_part[-1] - first_part[0]
            stable_part_transformed =  first_part + [val + diff for val in second_part]

            # subtract map
            stable_part_map = 1/3 * (np.max(elt['part']) - np.min(elt['part'])) + np.min(elt['part'])
            stable_part_minus_map = [stable_part_transformed[i] - stable_part_map for i in range(len(elt['part']))]
    
    res_c, res_r, res_s = [], [], []
    res_c_color, res_r_color, res_s_color = [], [], []
    min_first_part_len = 427
    for elt in data:
        if round(elt['params'][1], 2) == 1 and round(elt['params'][2], 2) == 1:

            part = elt['part']
            # transform
            first_part, second_part = elt['part'][np.argmin(elt['part']):], elt['part'][:np.argmin(elt['part'])]
            diff = second_part[-1] - first_part[0]
            part_transformed =  first_part + [val + diff for val in second_part]
            part_transformed_map = 1/3 * (np.max(elt['part']) - np.min(elt['part'])) + np.min(elt['part'])
            part_transformed_minus_map = [part_transformed[i] - part_transformed_map for i in range(len(elt['part']))]

            if len(first_part) <= min_first_part_len:
                min_first_part_len = len(first_part)

            # normalize both
            part_transformed_minus_map_normalized = [(part_transformed_minus_map[i] - np.min(part_transformed_minus_map)) / (np.max(part_transformed_minus_map)-np.min(part_transformed_minus_map)) for i in range(len(elt['part']))]
            stable_part_minus_map_normalized = [(stable_part_minus_map[i] - np.min(stable_part_minus_map)) / (np.max(stable_part_minus_map)-np.min(stable_part_minus_map)) for i in range(len(elt['part']))]

            res = [part_transformed_minus_map_normalized[i] - stable_part_minus_map_normalized[i] for i in range(len(elt['part']))]
            res_c.append(res)
            res_c_color.append(round(part_transformed_map/5)*5)
            
        elif round(elt['params'][0], 2) == 1 and round(elt['params'][2], 2) == 1:
            part = elt['part']
            # transform
            first_part, second_part = elt['part'][np.argmin(elt['part']):], elt['part'][:np.argmin(elt['part'])]
            diff = second_part[-1] - first_part[0]
            part_transformed =  first_part + [val + diff for val in second_part]
            part_transformed_map = 1/3 * (np.max(elt['part']) - np.min(elt['part'])) + np.min(elt['part'])
            part_transformed_minus_map = [part_transformed[i] - part_transformed_map for i in range(len(elt['part']))]

            if len(first_part) <= min_first_part_len:
                min_first_part_len = len(first_part)

            # normalize both
            part_transformed_minus_map_normalized = [(part_transformed_minus_map[i] - np.min(part_transformed_minus_map)) / (np.max(part_transformed_minus_map)-np.min(part_transformed_minus_map)) for i in range(len(elt['part']))]
            stable_part_minus_map_normalized = [(stable_part_minus_map[i] - np.min(stable_part_minus_map)) / (np.max(stable_part_minus_map)-np.min(stable_part_minus_map)) for i in range(len(elt['part']))]

            res = [part_transformed_minus_map_normalized[i] - stable_part_minus_map_normalized[i] for i in range(len(elt['part']))]
            res_r.append(res)
            res_r_color.append(round(part_transformed_map/5)*5)
        elif round(elt['params'][0], 2) == 1 and round(elt['params'][1], 2) == 1:
            part = elt['part']
            # transform
            first_part, second_part = elt['part'][np.argmin(elt['part']):], elt['part'][:np.argmin(elt['part'])]
            diff = second_part[-1] - first_part[0]
            part_transformed =  first_part + [val + diff for val in second_part]
            part_transformed_map = 1/3 * (np.max(elt['part']) - np.min(elt['part'])) + np.min(elt['part'])
            part_transformed_minus_map = [part_transformed[i] - part_transformed_map for i in range(len(elt['part']))]

            if len(first_part) <= min_first_part_len:
                min_first_part_len = len(first_part)

            # normalize both
            part_transformed_minus_map_normalized = [(part_transformed_minus_map[i] - np.min(part_transformed_minus_map)) / (np.max(part_transformed_minus_map)-np.min(part_transformed_minus_map)) for i in range(len(elt['part']))]
            stable_part_minus_map_normalized = [(stable_part_minus_map[i] - np.min(stable_part_minus_map)) / (np.max(stable_part_minus_map)-np.min(stable_part_minus_map)) for i in range(len(elt['part']))]

            res = [part_transformed_minus_map_normalized[i] - stable_part_minus_map_normalized[i] for i in range(len(elt['part']))]
            res_s.append(res)
            res_s_color.append(round(part_transformed_map/5)*5)

    # make all plots only the lenght of min_first_part_len
    # res_c = [res[:min_first_part_len] for res in res_c]
    # res_r = [res[:min_first_part_len] for res in res_r]
    # res_s = [res[:min_first_part_len] for res in res_s]

    print(min(res_c_color), max(res_c_color))
    print(min(res_r_color), max(res_r_color))
    print(min(res_s_color), max(res_s_color))

    # make 3 side by side plots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Flatten the 2x3 grid of axes for easy iteration
    axs = axs.flatten()

    # add title for each subplot
    axs[0].set_title('Kontraktilnost')
    axs[1].set_title('Upornost')
    axs[2].set_title('Podajnost')

    
    for i, res in enumerate([res_c, res_r, res_s]):
        for j, elt in enumerate(res):
            if i == 2:
                x_axis = np.linspace(0, 1, len(elt))
                axs[i].plot(x_axis, elt, color=color_dict[res_s_color[j]])
            else:
                # x_axis should be of len(part) but from 0 to 0.85
                x_axis = np.linspace(0, 0.85, len(elt))
                axs[i].plot(x_axis, elt, color=color_dict[res_c_color[j]])

    # set names for x and y axis
    axs[0].set_xlabel('čas [s]', fontsize=15)
    axs[0].set_ylabel('relativna sprememba [mmHg]', fontsize=15)
    axs[1].set_xlabel('čas [s]', fontsize=15)
    axs[1].set_ylabel('relativna sprememba [mmHg]', fontsize=15)
    axs[2].set_xlabel('normaliziran čas', fontsize=15)
    axs[2].set_ylabel('normalizirana relativna sprememba', fontsize=15)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the plot
    plt.show()


colors = {
    55: '#ffffcc',  # very light yellow
    60: '#ffffb2',  # light yellow
    65: '#fed976',  # yellow-orange
    70: '#feb24c',  # orange
    75: '#fd8d3c',  # darker orange
    80: '#fc4e2a',  # red-orange
    85: '#e31a1c',  # red
    90: '#bd0026',  # darker red
    95: '#800026',  # even darker red
    100: '#660000', # deep red
    105: '#4c0000', # very deep red
    110: '#330000', # darkest red
    115: '#1a0000'  # near black
}




if __name__ == "__main__":

    # seznam slovarjev s podatki
    # data = read_new_data()
    # print("Število simulacij (do zdaj):", len(data))

    # data = filter_data(data)
    # print("Število simulacij (po filtriranju):", len(data))

    # plot_3d(data)

    # data = read_parsed_data()	
    # # plot_all_simulated_ponts(data)
    # # plot_death_points(data)
    # plot_2d(data)

    #====================================================
    data = read_parsed_data()

    # print(len(data))
    # new_data = []
    # for elt in data:
    #     # if elt['part'] doesnt contain any nan values and any negative values and has at least two different values add it to new_data
    #     if not np.isnan(elt['part']).any() and np.min(elt['part']) >= 0 and len(set(elt['part'])) != 1:
    #         new_data.append(elt)
    # print(len(new_data))

    print('plotting')

    # plot_all_simulated_ponts(data)
    # plot_death_points(new_data, map_threshold=65, flow_threshold=0.00005, map_upper_threshold=100)
    # 46 prenizek flow
    # 37 nad 150 map

    # plot_death_points(data)
    # plot_flow_and_pp(data)
    # plot_death_points_side_by_side(data)

    # plot_just_one_wave(data)
    # plot_2d(data)

    plot_2d_res(data)

    # for elt in new_data:
    #     if round(elt['params'][0], 2) == 1.95 and round(elt['params'][1], 2) == 1.35 and round(elt['params'][2], 2) == 0.1: 
    #         plt.plot(elt['part'])
    #         plt.show()

    # plot_main_points(data)

    # plot_decomposition(data, color_dict=colors)

    # plot_single_wave_for_features(data)
