# ucitavanje podataka i iscrtavanje
import pandas
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


# ucitavanje podataka

def load_data():
    data = pandas.read_csv("C:\\Users\\teodo\Desktop\ori\credit_card_clustering\data\credit_card_data.csv")
    #data = pandas.read_csv("../data/credit_card_data.csv")
    data = data.fillna(data.median())
    vals = data.iloc[:, 1:].values
    return data, vals


# odredjivanje broja klastera

def number_cluster(vals):
    wcss = []
    for ii in range(1, 12):
        kmeans = KMeans(n_clusters=ii, init="k-means++", n_init=10, max_iter=300)
        kmeans.fit_predict(vals)
        wcss.append( kmeans.inertia_ )
    plt.plot(wcss, 'ro-', label="WCSS")
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.show()

def kmeans_function(data, vals):
    kmeans = KMeans(n_clusters=10, init="k-means++", n_init=10, max_iter=300)
    y_pred = kmeans.fit_predict(vals)
    # for i in y_pred:
    #     print(i)
    # print(len(y_pred))
    data["cluster"] = y_pred
    cols = list(data.columns)
    cols.remove("CUST_ID")

    # sns.pairplot(data[cols], hue="cluster")
    # plt.show()


if __name__ == '__main__':
    data, vals = load_data()
    #number_cluster(vals)
    # sa crteza uzimamo da je broj klastera 10
    k = 10
    kmeans_function(data, vals)



