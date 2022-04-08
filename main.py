from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as ps
import numpy as np
import matplotlib.pyplot as plt


def wiPCA(tablica_wartosci):
    tablica_usredniona = tablica_wartosci - np.mean(tablica_wartosci, axis=0)   # Uśredniamy wartości tablicy

    return


tablica = np.random.rand(20, 20)
tablica = tablica.reshape(200, 2)
plt.scatter(tablica[:, [0]], tablica[:, [1]])
plt.show()

# Funkcja wbudowana
pca = PCA(n_components=1)
f_wbudowana = pca.fit(tablica).transform(tablica)
print(f_wbudowana)
plt.scatter(f_wbudowana, f_wbudowana)
plt.show()
