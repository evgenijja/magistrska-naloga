import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline
import json 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

def get_one_wave():

    with open("data/parsed_data2.json", "r") as json_file:
        data = json.load(json_file)

    new_data = []
    for point in data:
        params = point['params']
        if round(params[0], 2) == 1.0 and round(params[1], 2) == 1.0 and round(params[2], 2) == 1.0:
            stable_part = point['part']

            stable_part = stable_part[np.argmin(stable_part):] + stable_part[:np.argmin(stable_part)]
            return stable_part
        

def scree_plot(filepath="feature_definition/pca/pca_results.csv"):
    df = pd.read_csv(filepath)
    y1, y2 = df["prop_explained"], df["cumsum.prop_explained."]

    y1, y2 = y1[:20], y2[:20] # vidim da so od tam naprej sam še enke

    x = list(np.arange(len(y1)+1))[1:]

    plt.plot(x, y1, 'orange', label='Razložena varianca', marker='o')
    plt.plot(x, y2, 'g', label='Kumulativna razložena varianca', marker='o')

    # add annotation to first 3 points
    for i in range(2):
        plt.text(x[i], y1[i], f"{y1[i]*100:.2f} %")
        plt.text(x[i], y2[i], f"{y2[i]*100:.2f} %")

    # show x ticks that are in x
    plt.xticks(x)

    # add axis labels
    plt.xlabel("Glavne komponente")
    plt.ylabel("Delež razložene variance")

    plt.legend()
    plt.show()


def plot_pca_results_znacilke(filepath="feature_definition/pca/loadings_results2_znacilke.csv"):

    df = pd.read_csv(filepath)

    y1, y2, y3 = df["PC1"][1:], df["PC2"][1:], df["PC3"][1:]
    x = list(np.arange(len(y1)))

    # make bar plot with two bars for each class, we have len(y1) classes and y1 and y2 are the values for each class
    # Width of bars
    width = 0.35

    # Position of bars on the x-axis
    x_positions = np.arange(len(x))

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(x_positions - width/2, y1, width=width, label='Y1')  # Bars for y1
    plt.bar(x_positions + width/2, y2, width=width, label='Y2')  # Bars for y2

    # Add labels and title
    plt.xlabel('Classes (X)')
    plt.ylabel('Values')
    plt.title('Grouped Bar Plot for Y1 and Y2 by Classes')
    plt.xticks(x_positions, x, rotation=90)  # Set x-ticks as class values, rotate for readability

    # Add a legend
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_pca_results(filepath="feature_definition/pca/normalized_pca_results_10092024_2.csv", stable_part=None):
    df = pd.read_csv(filepath)

    y1, y2, y3 = df["PC1"][1:], df["PC2"][1:], df["PC3"][1:]
    x = list(np.arange(len(y1)))
    const = len(y1) * [0]

    # Create the main plot
    fig, ax1 = plt.subplots()

    # Plot y1 (PC1), y2 (PC2), and y3 (PC3) on the primary axis
    ax1.plot(x, y1, 'r', label='PC1')
    ax1.plot(x, y2, 'g', label='PC2')
    # ax1.plot(x, y3, color='orange', label='PC3')
    ax1.plot(x, const, 'k--')

    # Set labels for the primary y-axis
    ax1.set_ylabel('Vrednost uteži')

    ax1.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 426])  # Set ticks at regular intervals
    # modify ticks so that they show the number that is i + 1
    ax1.set_xticklabels([str(i+1) for i in [0, 49, 99, 149, 199, 249, 299, 349, 399, 426]])
    # ax1.set_xticks([0, len(x)-1])  # Ensure the first and last tick are shown


    # If stable_part is provided, create the secondary y-axis
    if stable_part is not None:
        ax2 = ax1.twinx()  # create a twin y-axis
        ax2.plot(x, stable_part[1:], color='lightgray', linestyle='--', label='Referenčni val')
        ax2.set_ylabel('Referenčni val')

        # Gather handles and labels for both axes for the legend
        lines_1, labels_1 = ax1.get_legend_handles_labels()  # Get legend info from primary y-axis
        lines_2, labels_2 = ax2.get_legend_handles_labels()  # Get legend info from secondary y-axis
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')  # Combine and plot the legends

    else:
        # Plot the legend for the primary y-axis only
        ax1.legend(loc='best')

    plt.show()

def make_biplot_for_features(loadings_path="feature_definition/pca/loadings_results.csv", znacilke=False):

    loadings = pd.read_csv(loadings_path)
    # scores = pd.read_csv(scores_path)

    # create 427 colors from light blue to dark blue
    cmap = plt.cm.get_cmap('Blues', 1000)

    # make two side by side plots
    fig, axs = plt.subplots()

    # plot loadings
    pc1, pc2 = loadings["PC1"], loadings["PC2"]

    if not znacilke:
        for i in range(len(pc1)):
            axs.scatter(pc1[i], pc2[i], color=cmap(i+350))

        # add label to each 50th point
        for i in range(0, len(pc1), 50):
            if i != 0:
                axs.text(pc1[i], pc2[i], f"{i}")
        # axs.text(pc1[426], pc2[426], f"{427}")

    else: 
        g1, g2, g3, g4, g5, g6 = [], [], [], [], [], []
        for i in range(len(pc1)):
            # axs.scatter(pc1[i], pc2[i], color=cmap(i*5+350))
            # axs.scatter(pc1[i], pc2[i], color='blue')

            x, y = pc1[i], pc2[i]
            if x > 0.15 and y > 0:
                axs.scatter(pc1[i], pc2[i], color='blue')
                g1.append(i)

            elif x > 0.1 and y < - 0.01:
                axs.scatter(pc1[i], pc2[i], color='red')
                g2.append(i)

            elif x > 0 and x < 0.1 and y < -0.24:
                g3.append(i)
                axs.scatter(pc1[i], pc2[i], color='green')

            elif x < 0 and x > -0.1 and y < -0.2:
                axs.scatter(pc1[i], pc2[i], color='yellow')
                g4.append(i)

            elif x > -0.2 and x < -0.1 and y > - 0.2 and y < -0.1:
                axs.scatter(pc1[i], pc2[i], color='purple')
                g5.append(i)

            elif x > -0.2 and x < -0.1 and y > - 0.1 and y < 0:
                axs.scatter(pc1[i], pc2[i], color='orange')
                g6.append(i)

            else:
                axs.scatter(pc1[i], pc2[i], color='black')

        # add label to each 10th point
        # for i in range(0, len(pc1), 10):
        #     if i != 0:
        #         axs.text(pc1[i], pc2[i], f"{i}")

        # for i in range(0, len(pc1)):
        #     axs.text(pc1[i], pc2[i], f"{i}", fontsize=15)

    print("BLUE")
    print(g1)
    print("=====================================")
    print("RED")
    print(g2)
    print("=====================================")
    print("GREEN")
    print(g3)
    print("=====================================")
    print("YELLOW")
    print(g4)
    print("=====================================")
    print("PURPLE")
    print(g5)
    print("=====================================")
    print("ORANGE")
    print(g6)


    # set x and y axis labels
    axs.set_xlabel("PC1", fontsize=18)
    axs.set_ylabel("PC2", fontsize=18)

    # add title
    axs.set_title("Uteži", fontsize=18)

    plt.show()

def feature_correlation_matrix(filepath="feature_definition/pca/scaled_features.csv"):

    features = pd.read_csv(filepath)

    # calculate correlation matrix
    corr_matrix = features.corr()

    # plot the correlation matrix
    fig, ax = plt.subplots(figsize=(20, 20))
    cax = ax.matshow(corr_matrix, cmap='coolwarm')

    # set axis labels as numbers of rows and columns and rotate x tick for 90 degrees
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    # ax.set_xticklabels(corr_matrix.columns, rotation=90)

    plt.xticks(rotation=90)

    # add colorbar
    fig.colorbar(cax)

    # plt.savefig() 
    plt.show()

def feature_correlation_matrix2(filepath="feature_definition/pca/scaled_features.csv"):

    features = pd.read_csv(filepath)

    # do not include the first three columns
    features = features.iloc[:, 4:]

    # calculate correlation matrix
    cor_matrix = features.corr()
    # cor_matrix = pd.DataFrame(features.corr())
    print(cor_matrix.shape)

    # cor_matrix = np.corrcoef(features)  # Compute correlations

    # # Convert correlation to distance matrix
    # # Distance = 1 - Correlation
    # dist_matrix = 1 - cor_matrix
    # # dist_matrix = np.fill_diagonal(dist_matrix, 0)
    # # dist_matrix = (dist_matrix + dist_matrix.T) / 2 
    # # Ensure the distance matrix is symmetric and convert to condensed form
    # condensed_dist_matrix = squareform(dist_matrix)

    feature_mapping = {f: i for i, f in enumerate(cor_matrix.columns)}
    print(feature_mapping)
    cor_matrix = cor_matrix.rename(index=feature_mapping, columns=feature_mapping)

    # Perform hierarchical clustering
    linkage_matrix = linkage(cor_matrix, method='average') 

    # Create heatmap with clustering
    sns.clustermap(cor_matrix, 
                row_linkage=linkage_matrix, 
                col_linkage=linkage_matrix, 
                cmap="coolwarm", 
                # annot=True, 
                figsize=(8, 8),
                # xticklabels=True,  # Ensure column labels are shown
                yticklabels=True,
                # set size for ticklabels
                cbar=False

                )
                
    # plt.setp(plt.gca().get_xticklabels(), fontsize=18) 
    # hide colorbar
    # plt.colorbar().remove()
    # plt.title("Heatmap with Clustering")
    # tight layout

    # make the colorbar smaller


    plt.tight_layout()
    plt.show()



def make_biplot(filepath, color="quadrant", which="tocke"):

    scores = pd.read_csv(filepath)
    pc1, pc2 = scores["PC1"], scores["PC2"]

    # add another column to scores that depend on the values in columns C, R, S
    # if c < 1 and r < 1 and s < 1, color = 1
    # scores["color"] = scores.apply(lambda row: 1 if row["C"] <= 1 and row["R"] <= 1 and row["S"] <= 1 else 0, axis=1)
    # scores["color"] = scores.apply(lambda row: 2 if row["C"] <= 1 and row["R"] <= 1 and row["S"] >= 1 else row["color"], axis=1)
    # scores["color"] = scores.apply(lambda row: 3 if row["C"] <= 1 and row["R"] >= 1 and row["S"] <= 1 else row["color"], axis=1)
    # scores["color"] = scores.apply(lambda row: 4 if row["C"] <= 1 and row["R"] >= 1 and row["S"] >= 1 else row["color"], axis=1)
    # scores["color"] = scores.apply(lambda row: 5 if row["C"] >= 1 and row["R"] <= 1 and row["S"] <= 1 else row["color"], axis=1)
    # scores["color"] = scores.apply(lambda row: 6 if row["C"] >= 1 and row["R"] <= 1 and row["S"] >= 1 else row["color"], axis=1)
    # scores["color"] = scores.apply(lambda row: 7 if row["C"] >= 1 and row["R"] >= 1 and row["S"] <= 1 else row["color"], axis=1)
    # scores["color"] = scores.apply(lambda row: 8 if row["C"] >= 1 and row["R"] >= 1 and row["S"] >= 1 else row["color"], axis=1)

    scores["color"] = scores.apply(lambda row: 1 if row["C"] <= 1 and row["R"] <= 1 else 0, axis=1)
    # scores["color"] = scores.apply(lambda row: 2 if row["C"] <= 1 and row["R"] <= 1 else row["color"], axis=1)
    scores["color"] = scores.apply(lambda row: 2 if row["C"] <= 1 and row["R"] >= 1 else row["color"], axis=1)
    # scores["color"] = scores.apply(lambda row: 4 if row["C"] <= 1 and row["R"] >= 1 else row["color"], axis=1)
    scores["color"] = scores.apply(lambda row: 3 if row["C"] >= 1 and row["R"] <= 1 else row["color"], axis=1)
    # scores["color"] = scores.apply(lambda row: 6 if row["C"] >= 1 and row["R"] <= 1 else row["color"], axis=1)
    scores["color"] = scores.apply(lambda row: 4 if row["C"] >= 1 and row["R"] >= 1 else row["color"], axis=1)
    # scores["color"] = scores.apply(lambda row: 8 if row["C"] >= 1 and row["R"] >= 1 else row["color"], axis=1)

    # take tab10 colors

   

    # color_dict = {0 : 'black', 1: 'red', 2: 'red', 3: 'red', 4: 'red', 5: 'red', 6: 'red', 7: 'red', 8: 'red'}

    tab10 = plt.cm.get_cmap('tab10', 10)
    color_dict = {0 : tab10(0), 1: tab10(0), 2: tab10(1), 3: tab10(2), 4: tab10(3), 5: tab10(5), 6: tab10(6), 7: tab10(7), 8: tab10(8)}

    # plot scatter plot of pc1 and pc2

    if color == "quadrant":


        # scatter and make the size of the points smaller and make the color a little transparetn
        plt.scatter(pc1, pc2, c=scores["color"].apply(lambda x: color_dict[x]), s=7.5)
        # plt.scatter(pc1, pc2, c=scores["color"].apply(lambda x: color_dict[x]))

        # plot dashed line that has slope 1
        # plt.plot([-1, 2], [2, -1], 'k--')


        plt.show()

        ind1, ind2 = [], []
        ind3 = []
        for i in range(len(pc1)):
            if pc1[i] > 20:
                ind3.append(i)
            if pc1[i] > 2:
                ind1.append(i)
            elif pc2[i] > 1:
                ind2.append(i)

        # print(ind1)
        # print(ind2)
        print(ind3)

    
    else:
        if which == "tocke":

            # scatter points with one color and add dashed lines
            # vertical at 0.5, horizontal at- 0.5, vertical at -0.5
            plt.scatter(pc1, pc2, c='blue', s=7.5)
            plt.plot([0.5, 0.5], [-1, 1.5], 'k--')
            plt.plot([-1.5, 1.5], [-0.5, -0.5], 'k--')
            plt.plot([-0.5, -0.5], [-1, 1.5], 'k--')

            plt.show() 


        if which == "znacilke":

            # vertical at 3, vertical at -10. horizontal at 0
            plt.scatter(pc1, pc2, c='blue', s=7.5)
            plt.plot([3, 3], [-8, 6], 'k--')
            plt.plot([-10, -10], [-8, 6], 'k--')
            plt.plot([-20, 20], [0, 0], 'k--')

            plt.show()



    

def make_biplot_for_scores_tocke(scores_path="feature_definition/pca/scores_results2.csv"):

    scores = pd.read_csv(scores_path)

    pc1, pc2 = scores["PC1"], scores["PC2"]

    # oc1 večja od 2

    # plot scatter plot of pc1 and pc2
    plt.scatter(pc1, pc2)

    # set axis labels
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    # title
    plt.title("Valovi v novem koordinatnem sistemu")

    plt.show()

def make_biplot_for_scores_znacilke(scores_path="feature_definition/pca/scores_results2.csv"):

    scores = pd.read_csv(scores_path)

    pc1, pc2 = scores["PC1"], scores["PC2"]

    # oc1 večja od 2

    # plot scatter plot of pc1 and pc2
    plt.scatter(pc1, pc2)

    # set axis labels
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    # title
    plt.title("Valovi v novem koordinatnem sistemu")

    plt.show()



def plot_t_sne(filepath, color="quadrant"):

    tsne = pd.read_csv(filepath)

    pc1, pc2 = tsne["1"], tsne["2"]

    # add another column to tsne that depend on the values in columns C, R, S
    # if c < 1 and r < 1 and s < 1, color = 1
    tsne["color"] = tsne.apply(lambda row: 1 if row["C"] <= 1 and row["R"] <= 1 and row["S"] <= 1 else 0, axis=1)
    tsne["color"] = tsne.apply(lambda row: 2 if row["C"] <= 1 and row["R"] <= 1 and row["S"] >= 1 else row["color"], axis=1)
    tsne["color"] = tsne.apply(lambda row: 3 if row["C"] <= 1 and row["R"] >= 1 and row["S"] <= 1 else row["color"], axis=1)
    tsne["color"] = tsne.apply(lambda row: 4 if row["C"] <= 1 and row["R"] >= 1 and row["S"] >= 1 else row["color"], axis=1)
    tsne["color"] = tsne.apply(lambda row: 5 if row["C"] >= 1 and row["R"] <= 1 and row["S"] <= 1 else row["color"], axis=1)
    tsne["color"] = tsne.apply(lambda row: 6 if row["C"] >= 1 and row["R"] <= 1 and row["S"] >= 1 else row["color"], axis=1)
    tsne["color"] = tsne.apply(lambda row: 7 if row["C"] >= 1 and row["R"] >= 1 and row["S"] <= 1 else row["color"], axis=1)
    tsne["color"] = tsne.apply(lambda row: 8 if row["C"] >= 1 and row["R"] >= 1 and row["S"] >= 1 else row["color"], axis=1)

    # take tab10 colors

   

    color_dict = {0 : 'black', 1: 'red', 2: 'blue', 3: 'green', 4: 'yellow', 5: 'purple', 6: 'orange', 7: 'pink', 8: 'brown'}

    tab10 = plt.cm.get_cmap('tab10', 10)
    color_dict = {0 : tab10(0), 1: tab10(1), 2: tab10(2), 3: tab10(3), 4: tab10(4), 5: tab10(5), 6: tab10(6), 7: tab10(7), 8: tab10(8)}

    # plot scatter plot of pc1 and pc2

    if color == "quadrant":


        # scatter and make the size of the points smaller and make the color a little transparetn
        plt.scatter(pc1, pc2, c=tsne["color"].apply(lambda x: color_dict[x]), s=10)
        # plt.scatter(pc1, pc2, c=scores["color"].apply(lambda x: color_dict[x]))

        # plot dashed line that has slope 1
        # plt.plot([-1, 2], [2, -1], 'k--')


        plt.show()

def get_waves_by_indexes(indexes):

    with open("data/parsed_data2.json", "r") as json_file:
        data = json.load(json_file)

    new_data = []
    for point in data:
        part = point['part']
        if not np.isnan(part).any() and len(set(part)) > 1 and sum([1 for elt in part if elt < 0]) == 0:
            new_data.append(point)

    to_plot = []
    x, y, z = [], [], []
    for i in indexes:
        to_plot.append(new_data[i]['part'])
        x.append(new_data[i]['params'][0])
        y.append(new_data[i]['params'][1])
        z.append(new_data[i]['params'][2])

    return to_plot, x, y, z

def get_waves_by_params(parameter_set):

    to_plot = []

    with open("data/parsed_data2.json", "r") as json_file:
        data = json.load(json_file)

    for elt in data:
        params = elt['params']
        # print(params)

        if [round(params[0], 2), round(params[1], 2), round(params[2], 2)] in parameter_set:
            part = elt['part']
            min_index = np.argmin(part[:300])
            first_part, second_part = part[min_index:], part[:min_index]
            diff = second_part[0] - first_part[-1]

            part = part[min_index:] + [part[:min_index][j] - diff for j in range(len(part[:min_index]))]
            to_plot.append(part)

    # choose random 500 waves from to_plot if there are more than 500 else return all
    if len(to_plot) > 1000:
        return to_plot[:1000]
    else:   
        return to_plot

def plot_scores_subsets(filepath, mode="tocke"):

    scores = pd.read_csv(filepath)
    # print(scores.columns)
    pc1, pc2 = scores["PC1"], scores["PC2"]

    # add another column to scores that depend on the values in columns PC1 and PC2
    if mode == "tocke":
        scores["color"] = scores.apply(lambda row: 1 if row["PC1"] >= 0.5 else 0, axis=1)
        scores["color"] = scores.apply(lambda row: 2 if row["PC1"] < 0.5 and row["PC2"] >= -0.5 else row["color"], axis=1)
        scores["color"] = scores.apply(lambda row: 3 if row["PC2"] < -0.5  else row["color"], axis=1)

    elif mode == "znacilke":
        scores["color"] = scores.apply(lambda row: 1 if row["PC1"] >= 5 else 0, axis=1)
        scores["color"] = scores.apply(lambda row: 2 if row["PC1"] < 5 and row["PC2"] >= 0 else row["color"], axis=1)
        scores["color"] = scores.apply(lambda row: 3 if row["PC1"] < 5 and row["PC2"] < 0  else row["color"], axis=1)

    # take the rows where color is 1 and give me a list of lists of rows but only for columns C, R, S
    subset1 = scores[scores["color"] == 1][["C", "R", "S"]].values.tolist()
    subset2 = scores[scores["color"] == 2][["C", "R", "S"]].values.tolist()
    subset3 = scores[scores["color"] == 3][["C", "R", "S"]].values.tolist()

    print(len(subset1), len(subset2), len(subset3))
    print(subset1[0:10])

    subset1 = [[round(elt[0], 2), round(elt[1], 2), round(elt[2], 2)] for elt in subset1]
    subset2 = [[round(elt[0], 2), round(elt[1], 2), round(elt[2], 2)] for elt in subset2]
    subset3 = [[round(elt[0], 2), round(elt[1], 2), round(elt[2], 2)] for elt in subset3]

    to_plot1 = get_waves_by_params(subset1)
    to_plot2 = get_waves_by_params(subset2)
    to_plot3 = get_waves_by_params(subset3)

    print(len(to_plot1), len(to_plot2), len(to_plot3))

    # plot 2 rows 2 columns of plots - on the first one make a scatter plot of pc1 and pc2 and color them by color
    # on the rest of the plots make line plots of the waves, color them by color

    color_dict = {1: 'red', 2: 'blue', 3: 'green'}

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # scatter plot of pc1 and pc2
    axs[0, 0].scatter(pc1, pc2, c=scores["color"].apply(lambda x: color_dict[x]), s=10)

    # plot the waves
    for i in range(len(to_plot1)):
        axs[0, 1].plot(to_plot1[i], color='red')
    for i in range(len(to_plot2)):
        axs[1, 0].plot(to_plot2[i], color='blue')
    for i in range(len(to_plot3)):
        axs[1, 1].plot(to_plot3[i], color='green')

    plt.show()


def plot_loadings_znacilke(filepath="feature_definition/pca/loadings_results2_znacilke.csv"):

    loadings = pd.read_csv(filepath)

    y1, y2, y3 = loadings["PC1"][1:], loadings["PC2"][1:], loadings["PC3"][1:]

    print(y1[:10])

    y1, y2 = list(y1), list(y2)

    pts = [[y1[i], y2[i]] for i in range(len(y1))]
    distances = [np.linalg.norm(pts[i]) for i in range(len(pts))]
    # sort pts by their distance from the origin in descneding order and take the first 10

    sorted_pts = [x for _, x in sorted(zip(distances, pts), reverse=True)][:10]

    # plot points but color them by their distance from the origin and add labels to the first 10 points 
    fig, ax = plt.subplots()
    for i in range(len(pts)):
        print(f"Znacilka: {i}")
        if pts[i] in sorted_pts:
            ax.text(pts[i][0], pts[i][1], f"{i}")
            print(f"Ful pomembna znacilka: {i}")
            ax.scatter(pts[i][0], pts[i][1], c='red')
        else:
            ax.scatter(pts[i][0], pts[i][1], c='blue')

    # for i in range(10):
    #     ax.text(pts[i][0], pts[i][1], f"{i}")

    plt.show()




def plot_waves_by_indexes(indexes, only_2d=False):

    with open("data/parsed_data2.json", "r") as json_file:
        data = json.load(json_file)

    new_data = []
    for point in data:
        part = point['part']
        if not np.isnan(part).any() and len(set(part)) > 1 and sum([1 for elt in part if elt < 0]) == 0:
            new_data.append(point)

    to_plot = []
    x, y, z = [], [], []
    for i in indexes:
        to_plot.append(new_data[i]['part'])
        x.append(new_data[i]['params'][0])
        y.append(new_data[i]['params'][1])
        z.append(new_data[i]['params'][2])
    
    # plot 2 side by side plots: the first one is 2d plot of to_plot 
    # the second one is 3d plot of x, y, z

    if only_2d:
        fig, axs = plt.subplots(1, 1, figsize=(10, 5))
        for i in range(len(to_plot)):
            part = to_plot[i]
            min_index = np.argmin(part)
            if min_index == 426 or min_index == 427:
                min_index = np.argmin(part[:round(0.5*len(part))])

            first_part, second_part = part[min_index:], part[:min_index]
            diff = second_part[0] - first_part[-1]
            part = part[min_index:] + [part[:min_index][j] - diff for j in range(len(part[:min_index]))]
            # first_part, second_part = part[np.argmin(part):] + part[:np.argmin(part)]
            plt.plot(part, c='red')
        plt.show()
        return

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    for elt in to_plot:
        print(np.argmin(elt))

    for i in range(len(to_plot)):
        part = to_plot[i]
        min_index = np.argmin(part)
        if min_index == 426 or min_index == 427:
            min_index = np.argmin(part[:round(0.5*len(part))])

        first_part, second_part = part[min_index:], part[:min_index]
        diff = second_part[0] - first_part[-1]
        part = part[min_index:] + [part[:min_index][j] - diff for j in range(len(part[:min_index]))]
        # first_part, second_part = part[np.argmin(part):] + part[:np.argmin(part)]
        axs[0].plot(part, c='red')

    # scatter in 3d
    axs[1] = fig.add_subplot(122, projection='3d')
    axs[1].scatter(x, y, z)

    # set axis labels
    axs[1].set_xlabel("c")
    axs[1].set_ylabel("r")
    axs[1].set_zlabel("s")

    # set fixed ranges
    axs[1].set_xlim(0.55, 2.15)
    axs[1].set_ylim(0.7, 1.5)
    axs[1].set_zlim(0.1, 1.3)

    plt.show()




if __name__=="__main__":

    # stable_part = get_one_wave()
    
    # plot_pca_results(filepath="feature_definition/pca/loadings_results2.csv", stable_part=stable_part)
    # scree_plot(filepath="feature_definition/pca/pca_results2.csv")

    # loadings1
    # make_biplot_for_features(loadings_path="feature_definition/pca/loadings_results2.csv")
    # make_biplot_for_scores_tocke(scores_path="feature_definition/pca/scores_results2.csv")


    # plot_pca_results_znacilke(filepath="feature_definition/pca/loadings_results2_znacilke.csv")
    # scree_plot(filepath="feature_definition/pca/pca_results2_znacilke.csv")

    # make_biplot_for_features(loadings_path="feature_definition/pca/loadings_results2_znacilke.csv", znacilke=True)
    # make_biplot_for_scores_tocke(scores_path="feature_definition/pca/scores_results2_znacilke.csv")

    # make_biplot("feature_definition/pca/scores_with_params_results2_znacilke.csv", color="quadrant")
    # make_biplot("feature_definition/pca/scores_with_params_results2_znacilke.csv", color="tocke", which="znacilke")

    # plot_t_sne("feature_definition/pca/tsne_tocke_with_params.csv", color="quadrant")

    # plot_scores_subsets("feature_definition/pca/scores_with_params_results2.csv", mode="tocke")

    # plot_loadings_znacilke(filepath="feature_definition/pca/loadings_results2_znacilke.csv")

    # features = pd.read_csv("feature_definition/pca/features_FINAL.csv")

    # for i in range(len(features.columns[4:])):
    #     if i+4 in [0, 1, 4, 8, 26, 31, 36, 39, 40, 45, 46]:
    #         print(f"{i} - {features.columns[i+4]}")
            # print(i, features.columns[i+4])


    feature_correlation_matrix2(filepath="data/features_FINAL.csv")

    # ind1 = [11326, 11351, 11401, 11451, 12026, 12076, 12126, 12176, 12201, 12251, 12676, 12701, 12726, 12751, 12801, 12851, 12926, 12951, 13001, 13326, 13351, 13401, 13451, 13501]
    # ind2 = [11376, 11426, 11476, 12001, 12051, 12101, 12151, 12226, 12276, 12301, 12776, 12826, 12901, 12976, 13526]

    # ind3 = [11326, 11351, 11401, 11451, 12026, 12076, 12126, 12176, 12201, 12251, 12701, 12726, 12751, 12801, 12851, 12926, 12951, 13001, 13326, 13351, 13401, 13451, 13501, 13876, 13901, 13951, 14001, 14051]
    
    # plot_waves_by_indexes(ind1 + ind2, only_2d=True)
    # plot_waves_by_indexes(ind3, only_2d=True)

    # plot_waves_by_indexes(ind2)
    # plot_waves_by_indexes(ind3)