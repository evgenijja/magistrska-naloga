import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from reading_data import * 

from sklearn.decomposition import PCA

"""
Zanima nas v katerem delu vala se valovi med sabo najbolj razlikujejo.
V ta namen naredimo PCA na vseh 427 točkah, ki definirajo vsak val, da dobimo prvi dve glavni komponenti in upamo, da bosta pojasnili skoraj vso varianco.
Potem lahko zračunam korelacijski koeficient vseh točk v času z glavnima komponentama. 

To lahko naredim za vse valove ali pa za valove kjer je bil spremenjen samo 1 param, mogoč po dva??
"""

"""
PCA NOTES:
- PCA is a linear transformation that finds the "principal components" of the data
- It can be thought of as a method that rotates the dataset in a way such that the rotated features are statistically uncorrelated
- The first principal component has the largest variance, the second has the second largest, and so on
- The rotation is often done in such a way that the first principal component explains as much of the variance in the data as possible, and each succeeding component explains as much of the remaining variance as possible
- The amount of variance explained by each principal component is a fraction of the total variance in the data
- The fraction of variance explained by a particular principal component is equal to the eigenvalue corresponding to that component divided by the sum of all eigenvalues
- The eigenvectors of the covariance matrix of the data are the principal components
- The eigenvalues of this matrix indicate the variance of the data in the direction of the corresponding eigenvector
- The eigenvectors are orthogonal, that is, they are perpendicular to each other
- The eigenvectors are unit vectors, that is, their length is equal to 1
- The eigenvectors point in the direction of the maximum variance of the data

WICHTIG:
- Any matrix of orthonormal vectors (unit vectors that are orthogonal to one another) represents a rotation and/or reflection of the axes of a Euclidean space.


"""


def generate_X(data, option='all'):
    """
    Matrika X je narejena tako, da ima 427 stolpcev, v vsakem stolpcu je točka v času. 
    data je seznam seznamov, vsak seznam ena vrstica
    """
    df = pd.DataFrame(data)
    df.dropna(inplace=True)

    return df


def run_pca(df):
    """
    FUL POMEMBNO! NORMALIZIRAJ PODATKE!

    Izvede PCA na danem df.
    TODO parametri v PCA??
    TODO no. components

    ven rabim dobit:
        - nove komponente (vektorji ki so iste dolžine kot original df) da bom lahko računala korelacijski koeficient
        - koliko variance pojasni posamezna komponenta oz usaj lastne vrednosti katerim pripradajo
    """

    pca = PCA()

    pca.fit(df)

    transformed_data = pca.fit_transform(df)
    transformed_df = pd.DataFrame(transformed_data)

    # components.shape = (no. components, no. features)
    components = pca.components_ 

    # explained varience.shape = no. components
    explained_varience = pca.explained_variance_ratio_

    # compute loadings for the first two components
    # loadings = np.corrcoef(df.T, transformed_df.T)[:df.shape[1], df.shape[1]:]    
    # https://stats.stackexchange.com/questions/143905/loadings-vs-eigenvectors-in-pca-when-to-use-one-or-another
    # https://stats.stackexchange.com/questions/119746/what-is-the-proper-association-measure-of-a-variable-with-a-pca-component-on-a/119758#119758
    # Loadings so projecirane točke v nov koordinatni sistem, ki ga določajo komponente.
    # Točke so projecirane na komponente, ki so vektorji, ki so dolgi toliko kot je število featurejev.
    # dobiš torej vektor dolžine featurjev za vsako komponento (ki predstavlja projekcije točk na to komponento in je zato korelacija med točkami in komponento)
    # loadings = pca.components_.T * np.sqrt(pca.explained_variance_ratio_)
    print(df.shape)
    print(pca.components_.shape)
    loadings = np.dot(df, pca.components_.T) * np.sqrt(pca.explained_variance_ratio_)

    return transformed_df, loadings, explained_varience, components

def plot_correlation(loadings):
    """
    Vzame matriko loadingov, ki predstavljajo korelacijo - vsak stolpec je ena komponenta, vrstic pa mora bit tok k featurjev.

    Kakšne use plote hočem?
    1) za vse valove
    2) 
    """ 

    x = np.arange(loadings.shape[0])

    plt.figure(figsize=(10, 6))
    plt.plot(x, loadings[:, 0], label='PC 1')
    plt.plot(x, loadings[:, 1], label='PC 2')
    plt.xlabel('Time points')
    plt.ylabel('Correlation')
    plt.title('Correlation between first two PCs and original features')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__=="__main__":

    normal_part = get_normal_data(data)
    parts, parts_c, parts_r, parts_s = extract_parts(data)


    # normalized_parts = normalize_parts(parts)
    translated_parts = first_translation(parts_s)
    normalized_translated_parts = normalize_parts(translated_parts)


    # generate X
    df = generate_X(normalized_translated_parts)

    # run PCA
    transformed_df, loadings, explained_varience, components = run_pca(df)

    # plot correlations
    plot_correlation(loadings)