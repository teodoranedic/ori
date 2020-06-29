import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sea
from sklearn.tree import _tree, DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ucitavanje podataka 
def load_data():
    data = pd.read_csv("C:\\Users\\teodo\Desktop\ori\credit_card_clustering\data\credit_card_data.csv")
    #data = pandas.read_csv("../data/credit_card_data.csv")
    data = data.fillna(data.median())
    vals = data.iloc[:, 1:].values  # numpy ndarray
    data.drop(data.columns[[0]], axis=1, inplace=True)  # brisi prvu kolonu iz dataframe-a

    # sve kolone
    # 0,      1,          2,                3         4                     5
    #CUST_ID,BALANCE,BALANCE_FREQUENCY,PURCHASES,ONEOFF_PURCHASES,INSTALLMENTS_PURCHASES,
    #   6               7                       8                       9
    # CASH_ADVANCE,PURCHASES_FREQUENCY,ONEOFF_PURCHASES_FREQUENCY,PURCHASES_INSTALLMENTS_FREQUENCY,
    #  10                          11            12               13         14      15
    # CASH_ADVANCE_FREQUENCY,CASH_ADVANCE_TRX,PURCHASES_TRX,CREDIT_LIMIT,PAYMENTS,MINIMUM_PAYMENTS,
    #   16               17
    # PRC_FULL_PAYMENT,TENURE

    return data, vals


# odredjivanje broja klastera
def elbow_method(vals):
    sse = []
    for k in range(1, 15):
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300)
        kmeans.fit_predict(vals)
        sse.append(kmeans.inertia_)
    plt.plot(sse, 'ro-', label="SSE")
    plt.xlabel("Number of clusters")
    plt.ylabel("SSE")
    plt.show()


def dimensionality_reduction(data):
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('dim_reduction', PCA(n_components=2, random_state=0))  # 2d prikaz
    ])
    pp = pipeline.fit_transform(data)
    return pp


def kmeans_clustering(pc):
    kmeans_model = KMeans(n_clusters=7)  # ?? menjaaaaaj mozda
    y_pred = kmeans_model.fit_predict(pc)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    sea.scatterplot(x=pc[:, 0], y=pc[:, 1], hue=y_pred, palette='bright', ax=ax)
    ax.set(xlabel="pc1", ylabel="pc2", title="Credit card clustering result")
    ax.legend(title='cluster')
    plt.show()

    #     cluster0 = data.loc[data['cluster'] == 0]
    #     cluster1 = data.loc[data['cluster'] == 1]
    #     cluster2 = data.loc[data['cluster'] == 2]
    #     cluster3 = data.loc[data['cluster'] == 3]
    #     cluster4 = data.loc[data['cluster'] == 4]
    #     cluster5 = data.loc[data['cluster'] == 5]
    #     cluster6 = data.loc[data['cluster'] == 6]
    #     cluster7 = data.loc[data['cluster'] == 7]
    #     cluster8 = data.loc[data['cluster'] == 8]
    #     cluster9 = data.loc[data['cluster'] == 9]
    return y_pred


def get_class_rules(tree: DecisionTreeClassifier, feature_names: list):
    inner_tree: _tree.Tree = tree.tree_
    classes = tree.classes_
    class_rules_dict = dict()

    # rekurzivno se poziva za cvorove
    def tree_dfs(node_id=0, current_rule=[]):
        # feature[i] holds the feature to split on, for the internal node i.
        split_feature = inner_tree.feature[node_id]
        if split_feature != _tree.TREE_UNDEFINED:  # internal node
            name = feature_names[split_feature]
            threshold = inner_tree.threshold[node_id]
            # left child
            left_rule = current_rule + ["({} <= {})".format(name, threshold)]
            tree_dfs(inner_tree.children_left[node_id], left_rule)
            # right child
            right_rule = current_rule + ["({} > {})".format(name, threshold)]
            tree_dfs(inner_tree.children_right[node_id], right_rule)
        else:  # leaf
            dist = inner_tree.value[node_id][0]
            dist = dist / dist.sum()
            max_idx = dist.argmax()
            if len(current_rule) == 0:
                rule_string = "ALL"
            else:
                rule_string = " and ".join(current_rule)
            # register new rule to dictionary
            selected_class = classes[max_idx]
            class_probability = dist[max_idx]
            class_rules = class_rules_dict.get(selected_class, [])
            class_rules.append((rule_string, class_probability))
            class_rules_dict[selected_class] = class_rules

    tree_dfs()  # start from root, node_id = 0
    return class_rules_dict


def cluster_report(data: pd.DataFrame, clusters, min_samples_leaf=50, pruning_level=0.01):
    # Create Model
    tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, ccp_alpha=pruning_level)
    tree.fit(data, clusters)

    # Generate Report
    feature_names = data.columns
    class_rule_dict = get_class_rules(tree, feature_names)

    report_class_list = []
    for class_name in class_rule_dict.keys():
        rule_list = class_rule_dict[class_name]
        combined_string = ""
        for rule in rule_list:
            combined_string += "[{}] {}\n\n".format(rule[1], rule[0])
        report_class_list.append((class_name, combined_string))

    cluster_instance_df = pd.Series(clusters).value_counts().reset_index()
    cluster_instance_df.columns = ['class_name', 'instance_count']
    report_df = pd.DataFrame(report_class_list, columns=['class_name', 'rule_list'])
    report_df = pd.merge(cluster_instance_df, report_df, on='class_name', how='left')
    print(report_df.to_string())


def pairplot():
    best_cols = ["PURCHASES","PAYMENTS","MINIMUM_PAYMENTS","BALANCE","CASH_ADVANCE", "CREDIT_LIMIT", "ONEOFF_PURCHASES"]
    best_vals = data[best_cols].iloc[:, :].values

    kmeans = KMeans(n_clusters=7, init="k-means++", n_init=10, max_iter=300)
    y_pred = kmeans.fit_predict(best_vals)

    data["cluster"] = y_pred
    data["hue"] = y_pred
    best_cols.append("cluster")
    best_cols.append("hue")
    sea.pairplot(data[best_cols], hue="hue")
    plt.show()


if __name__ == '__main__':
    data, vals = load_data()

    # scaling data
    pc = dimensionality_reduction(data)

    elbow_method(pc)

    # clustering
    y_cluster = kmeans_clustering(pc)

    # interpret results
    cluster_report(data, y_cluster, min_samples_leaf=50, pruning_level=0.001)

    pairplot()

