# ucitavanje podataka i iscrtavanje
import pandas
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# ucitavanje podataka

def load_data():
    data = pandas.read_csv("../data/credit_card_data.csv")
    data = data.fillna(data.median())
    return data


# odredjivanje broja klastera

def number_cluster(data):
    vals = data.iloc[:, 1:].values
    wcss = []
    for ii in range(1, 50):
        kmeans = KMeans(n_clusters=ii, init="k-means++", n_init=10, max_iter=300)
        kmeans.fit_predict( vals )
        wcss.append( kmeans.inertia_ )
    plt.plot(wcss, 'ro-', label="WCSS")
    plt.title("Computing WCSS for KMeans++")
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.show()

if __name__ == '__main__':
    data = load_data()

    number_cluster(data)
    # sa crteza uzimamo da je broj klastera 10
    k = 10

