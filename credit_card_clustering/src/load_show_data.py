# ucitavanje podataka i iscrtavanje
import pandas
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sea



# ucitavanje podataka

def load_data():
    data = pandas.read_csv("C:\\Users\\teodo\Desktop\ori\credit_card_clustering\data\credit_card_data.csv")
    #data = pandas.read_csv("../data/credit_card_data.csv")
    data = data.fillna(data.median())
    vals = data.iloc[:, 1:].values
    return data, vals


# odredjivanje broja klastera

def number_cluster(vals):
    sse = []
    for ii in range(1, 12):
        kmeans = KMeans(n_clusters=ii, init="k-means++", n_init=10, max_iter=300)
        kmeans.fit_predict(vals)
        sse.append( kmeans.inertia_ )
    plt.plot(sse, 'ro-', label="SSE")
    plt.xlabel("Number of clusters")
    plt.ylabel("SSE")
    plt.show()

def kmeans_function(data, vals):
    kmeans = KMeans(n_clusters=10, init="k-means++", n_init=10, max_iter=300)
    y_pred = kmeans.fit_predict(vals)
    data["cluster"] = y_pred
    #data["hue"] = y_pred
    cols = list(data.columns)
    # cols.remove("CUST_ID")

    cluster0 = data.loc[data['cluster'] == 0]
    cluster1 = data.loc[data['cluster'] == 1]
    cluster2 = data.loc[data['cluster'] == 2]
    cluster3 = data.loc[data['cluster'] == 3]
    cluster4 = data.loc[data['cluster'] == 4]
    cluster5 = data.loc[data['cluster'] == 5]
    cluster6 = data.loc[data['cluster'] == 6]
    cluster7 = data.loc[data['cluster'] == 7]
    cluster8 = data.loc[data['cluster'] == 8]
    cluster9 = data.loc[data['cluster'] == 9]

    # sea.pairplot(data[cols])
    # plt.show()
    return y_pred


if __name__ == '__main__':
    data, vals = load_data()
    #number_cluster(vals)
    # sa crteza uzimamo da je broj klastera 10
    k = 10

    # best_cols = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS"]
    # kmeans = KMeans(n_clusters=8, init="k-means++", n_init=10, max_iter=20)
    # best_vals = data[best_cols].iloc[:, 1:].values
    # y_pred = kmeans.fit_predict(best_vals)
    # data["hue"] = y_pred
    # best_cols.append("hue")
    # data["cluster"] = y_pred
    # best_cols.append("cluster")
    # sea.pairplot(data[best_cols], hue="hue")
    # plt.show()
    y_pred = kmeans_function(data, vals)




