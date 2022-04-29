from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as ps
import numpy as np
import matplotlib.pyplot as plt


def wiPCA(tablica_wartosci, zadany_wymiar=1):
    tablica_usredniona = tablica_wartosci - np.mean(tablica_wartosci, axis=0)   # Uśredniamy wartości tablicy

    # Obliczamy macierz kowariancji. Na przekątnej znajdują się wartości wariancji (ponieważ je uśredniliśmy)
    mat_kow = np.cov(tablica_usredniona, rowvar=False)

    # Obliczamy wartości własne macierzy kowariancji i wektor własnych. Im większe wartości własne,
    # tym większa różnorodność wartości
    wart_wlas_kow, wekt_wlas_kow = np.linalg.eig(mat_kow)

    # Pobieramy indeksy wartości własnych od największej do najmniejszej. Pozwolą nam posortować wektory
    # i wartości własne od najważniejszej do najmniej ważnej. Następnie sortujemy wartości i wektory własne według tych
    # indeksów
    indeksy_sortowania = np.argsort(wart_wlas_kow)[::-1]
    posort_wekt_wlas_kow = wekt_wlas_kow[:, indeksy_sortowania]

    # Wybieramy podzbiór z naszego wektora własnych. Liczba podzbiorów zależy od wymiaru, do którego chcemy zredukować
    # tablice
    liczba_podzbiorow = zadany_wymiar
    wekt_wlas_kow_podz = posort_wekt_wlas_kow[:, 0:liczba_podzbiorow]

    # Redukujemy tablicę do wymiaru podanego w zadanym parametrze
    tablica_wartosci_redukowana = np.dot(wekt_wlas_kow_podz.transpose(), tablica_usredniona.transpose()).transpose()

    # Kontrola wartości
    # print("Uśrednione wartości:", tablica_usredniona)
    # print("Wartości własne macierzy kowariancji: ", wart_wlas_kow)
    # print("Wektory własne macierzy kowariancji: ", wekt_wlas_kow)
    # print("Indeksy posortowane malejąco: ", indeksy_sortowania)
    # print("Posortowane wartości własne macierzy kowariancji: ", posort_wart_wlas_kow)
    # print("Posortowane wektory własne macierzy kowariancji: ", posort_wekt_wlas_kow)
    # print("Podzbiór wektora własnych: ", wekt_wlas_kow_podz)
    # print("Tablica zredukowana: ", tablica_wartosci_redukowana)
    return tablica_wartosci_redukowana, mat_kow


# Zad 1
tablica = np.random.rand(20, 20)
tablica = tablica.reshape(200, 2)
tablica_zredukowana, _ = wiPCA(tablica, zadany_wymiar=1)
plt.scatter(tablica[:, [0]], tablica[:, [1]])
plt.scatter(tablica_zredukowana, tablica_zredukowana)   # Podajemy tą samą tablicę, to wymiar jest 1D
plt.legend(['Wartości początkowe', 'Wartości zredukowane'])
plt.show()
# Funkcja wbudowana
pca = PCA(n_components=1)
f_wbudowana = pca.fit(tablica).transform(tablica)
plt.scatter(f_wbudowana, f_wbudowana)
plt.title('Wbudowana funkcja PCA')
plt.show()

# Zad 2
iris = datasets.load_iris()
data = iris.data
target = iris.target
iris_data_reduced, _ = wiPCA(data, zadany_wymiar=2)
plt.figure(figsize=(6, 6))
for c, i, tn in zip("rgb", [0, 1, 2], iris.target_names):
    plt.scatter(iris_data_reduced[target == i,0], iris_data_reduced[target == i,1], c=c, label=tn)
plt.legend()
plt.show()

# Zad 3
digits = datasets.load_digits()
digits_data_reduced, mat_kow_digits = wiPCA(digits.data, zadany_wymiar=2)
pca_digits = PCA().fit(digits.data)
plt.plot(np.cumsum(pca_digits.explained_variance_ratio_))
plt.show()

plt.scatter(digits_data_reduced[:, 0], digits_data_reduced[:, 1],
            c=digits.target, edgecolors='none', alpha=0.5, cmap=plt.cm.get_cmap('jet', 10))
plt.xlabel('składowa 1')
plt.ylabel('składowa 2')
plt.colorbar()
plt.show()

"""Krzywa kowariancji sposobem Kuby"""
"""Obliczamy wartości i wektory własne macierzy kowariancji, by obliczyć krzywą kowariancji"""
egn_values, egn_vectors = np.linalg.eig(mat_kow_digits)
total_egn_values = np.sum(egn_values)
var_exp = [(i / total_egn_values) for i in sorted(egn_values, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.plot(range(0, len(cum_var_exp)), cum_var_exp)
plt.show()
