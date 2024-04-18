import numpy as np
import pandas as pd

from reading_data import data, normal_part, parts

from sklearn.decomposition import PCA

"""
Zanima nas v katerem delu vala se valovi med sabo najbolj razlikujejo.
V ta namen naredimo PCA na vseh 427 točkah, ki definirajo vsak val, da dobimo prvi dve glavni komponenti in upamo, da bosta pojasnili skoraj vso varianco.
Potem lahko zračunam korelacijski koeficient vseh točk v času z glavnima komponentama. 

To lahko naredim za vse valove ali pa za valove kjer je bil spremenjen samo 1 param, mogoč po dva??
"""


def generate_X(data, option='all'):
    """
    Matrika X je narejena tako, da ima 427 stolpcev, v vsakem stolpcu je točka v času. 
    """
    df = pd.DataFrame(data)

    return df


def run_pca(df):
    """
    Izvede PCA na danem df.
    TODO parametri v PCA??
    TODO no. components

    ven rabim dobit:
        - nove komponente (vektorji ki so iste dolžine kot original df)
        - koliko variance pojasni posamezna komponenta oz usaj lastne vrednosti katerim pripradajo
    """

    pca = PCA()

    # Fit PCA on your data
    pca.fit(df)

    # Transform the original data into the new feature space
    transformed_data = pca.transform(df)

    # If you want to see the transformed data as a DataFrame
    transformed_df = pd.DataFrame(transformed_data)

    # Print the transformed data
    return transformed_df

if __name__=="__main__":
    df = generate_X(parts)