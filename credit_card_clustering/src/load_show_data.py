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

    # obrisi jos kolona koje nisu veoma bitne
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
    best_cols = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS", "PURCHASES_TRX"]
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

'''
   class_name  instance_count                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    rule_list
0           0            2838                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 [0.972007722007722] (PURCHASES_FREQUENCY <= 0.40833351016044617) and (CASH_ADVANCE_TRX <= 5.5) and (BALANCE <= 2888.3736572265625) and (PRC_FULL_PAYMENT <= 0.5357145071029663) and (CASH_ADVANCE <= 2067.3479614257812) and (PURCHASES <= 1320.3900146484375) and (BALANCE <= 1965.4468994140625)\n\n[0.9025974025974026] (PURCHASES_FREQUENCY <= 0.40833351016044617) and (CASH_ADVANCE_TRX <= 5.5) and (BALANCE <= 2888.3736572265625) and (PRC_FULL_PAYMENT <= 0.5357145071029663) and (CASH_ADVANCE <= 2067.3479614257812) and (PURCHASES <= 1320.3900146484375) and (BALANCE > 1965.4468994140625) and (CASH_ADVANCE_TRX <= 3.5)\n\n[0.6428571428571429] (PURCHASES_FREQUENCY <= 0.40833351016044617) and (CASH_ADVANCE_TRX <= 5.5) and (BALANCE <= 2888.3736572265625) and (PRC_FULL_PAYMENT <= 0.5357145071029663) and (CASH_ADVANCE > 2067.3479614257812) and (CREDIT_LIMIT <= 4250.0)\n\n[0.7627118644067796] (PURCHASES_FREQUENCY <= 0.40833351016044617) and (CASH_ADVANCE_TRX <= 5.5) and (BALANCE <= 2888.3736572265625) and (PRC_FULL_PAYMENT > 0.5357145071029663) and (PURCHASES_FREQUENCY <= 0.23611100018024445)\n\n[0.625] (PURCHASES_FREQUENCY <= 0.40833351016044617) and (CASH_ADVANCE_TRX <= 5.5) and (BALANCE > 2888.3736572265625) and (CASH_ADVANCE_FREQUENCY <= 0.08712099865078926) and (PAYMENTS <= 1238.6702270507812)\n\n[0.9411764705882353] (PURCHASES_FREQUENCY <= 0.40833351016044617) and (CASH_ADVANCE_TRX > 5.5) and (CASH_ADVANCE <= 6043.52734375) and (CREDIT_LIMIT <= 1750.0) and (CASH_ADVANCE_FREQUENCY <= 0.4365074932575226) and (PAYMENTS <= 909.5316162109375)\n\n[0.5490196078431373] (PURCHASES_FREQUENCY <= 0.40833351016044617) and (CASH_ADVANCE_TRX > 5.5) and (CASH_ADVANCE <= 6043.52734375) and (CREDIT_LIMIT <= 1750.0) and (CASH_ADVANCE_FREQUENCY <= 0.4365074932575226) and (PAYMENTS > 909.5316162109375)\n\n[0.825503355704698] (PURCHASES_FREQUENCY > 0.40833351016044617) and (PURCHASES <= 1410.7749633789062) and (CASH_ADVANCE <= 1172.3317260742188) and (PURCHASES_FREQUENCY <= 0.5773810148239136) and (PRC_FULL_PAYMENT <= 0.09545449912548065) and (PURCHASES <= 502.77000427246094) and (INSTALLMENTS_PURCHASES <= 217.80500030517578)\n\n[0.33695652173913043] (PURCHASES_FREQUENCY > 0.40833351016044617) and (PURCHASES <= 1410.7749633789062) and (CASH_ADVANCE > 1172.3317260742188) and (CASH_ADVANCE <= 4500.915283203125) and (CASH_ADVANCE_FREQUENCY <= 0.23611100018024445)\n\n
1           4            2702  [0.47560975609756095] (PURCHASES_FREQUENCY <= 0.40833351016044617) and (CASH_ADVANCE_TRX <= 5.5) and (BALANCE <= 2888.3736572265625) and (PRC_FULL_PAYMENT <= 0.5357145071029663) and (CASH_ADVANCE <= 2067.3479614257812) and (PURCHASES > 1320.3900146484375)\n\n[0.9466666666666667] (PURCHASES_FREQUENCY <= 0.40833351016044617) and (CASH_ADVANCE_TRX <= 5.5) and (BALANCE <= 2888.3736572265625) and (PRC_FULL_PAYMENT > 0.5357145071029663) and (PURCHASES_FREQUENCY > 0.23611100018024445)\n\n[0.5303030303030303] (PURCHASES_FREQUENCY > 0.40833351016044617) and (PURCHASES <= 1410.7749633789062) and (CASH_ADVANCE <= 1172.3317260742188) and (PURCHASES_FREQUENCY <= 0.5773810148239136) and (PRC_FULL_PAYMENT <= 0.09545449912548065) and (PURCHASES <= 502.77000427246094) and (INSTALLMENTS_PURCHASES > 217.80500030517578)\n\n[0.7299270072992701] (PURCHASES_FREQUENCY > 0.40833351016044617) and (PURCHASES <= 1410.7749633789062) and (CASH_ADVANCE <= 1172.3317260742188) and (PURCHASES_FREQUENCY <= 0.5773810148239136) and (PRC_FULL_PAYMENT <= 0.09545449912548065) and (PURCHASES > 502.77000427246094)\n\n[0.9182692307692307] (PURCHASES_FREQUENCY > 0.40833351016044617) and (PURCHASES <= 1410.7749633789062) and (CASH_ADVANCE <= 1172.3317260742188) and (PURCHASES_FREQUENCY <= 0.5773810148239136) and (PRC_FULL_PAYMENT > 0.09545449912548065)\n\n[0.968129885748647] (PURCHASES_FREQUENCY > 0.40833351016044617) and (PURCHASES <= 1410.7749633789062) and (CASH_ADVANCE <= 1172.3317260742188) and (PURCHASES_FREQUENCY > 0.5773810148239136) and (BALANCE <= 3233.2625732421875) and (PURCHASES <= 1066.5900268554688) and (CASH_ADVANCE_FREQUENCY <= 0.17424249649047852) and (CREDIT_LIMIT <= 9750.0)\n\n[0.6666666666666666] (PURCHASES_FREQUENCY > 0.40833351016044617) and (PURCHASES <= 1410.7749633789062) and (CASH_ADVANCE <= 1172.3317260742188) and (PURCHASES_FREQUENCY > 0.5773810148239136) and (BALANCE <= 3233.2625732421875) and (PURCHASES <= 1066.5900268554688) and (CASH_ADVANCE_FREQUENCY <= 0.17424249649047852) and (CREDIT_LIMIT > 9750.0)\n\n[0.45263157894736844] (PURCHASES_FREQUENCY > 0.40833351016044617) and (PURCHASES <= 1410.7749633789062) and (CASH_ADVANCE <= 1172.3317260742188) and (PURCHASES_FREQUENCY > 0.5773810148239136) and (BALANCE <= 3233.2625732421875) and (PURCHASES <= 1066.5900268554688) and (CASH_ADVANCE_FREQUENCY > 0.17424249649047852)\n\n[0.9090909090909091] (PURCHASES_FREQUENCY > 0.40833351016044617) and (PURCHASES <= 1410.7749633789062) and (CASH_ADVANCE <= 1172.3317260742188) and (PURCHASES_FREQUENCY > 0.5773810148239136) and (BALANCE <= 3233.2625732421875) and (PURCHASES > 1066.5900268554688) and (PURCHASES_TRX <= 20.5)\n\n[0.660377358490566] (PURCHASES_FREQUENCY > 0.40833351016044617) and (PURCHASES <= 1410.7749633789062) and (CASH_ADVANCE <= 1172.3317260742188) and (PURCHASES_FREQUENCY > 0.5773810148239136) and (BALANCE <= 3233.2625732421875) and (PURCHASES > 1066.5900268554688) and (PURCHASES_TRX > 20.5) and (CREDIT_LIMIT <= 5850.0)\n\n[0.8762886597938144] (PURCHASES_FREQUENCY > 0.40833351016044617) and (PURCHASES > 1410.7749633789062) and (PURCHASES <= 5097.239990234375) and (PURCHASES_FREQUENCY <= 0.6833334863185883) and (BALANCE <= 2309.7713623046875) and (PURCHASES <= 2279.52001953125)\n\n[0.8333333333333334] (PURCHASES_FREQUENCY > 0.40833351016044617) and (PURCHASES > 1410.7749633789062) and (PURCHASES <= 5097.239990234375) and (PURCHASES_FREQUENCY > 0.6833334863185883) and (CASH_ADVANCE_TRX <= 12.5) and (PURCHASES_TRX <= 71.5) and (PURCHASES <= 1901.2849731445312) and (PURCHASES_TRX <= 24.5) and (CREDIT_LIMIT <= 4050.0)\n\n
2           5            1443                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        [0.5769230769230769] (PURCHASES_FREQUENCY <= 0.40833351016044617) and (CASH_ADVANCE_TRX <= 5.5) and (BALANCE <= 2888.3736572265625) and (PRC_FULL_PAYMENT <= 0.5357145071029663) and (CASH_ADVANCE <= 2067.3479614257812) and (PURCHASES <= 1320.3900146484375) and (BALANCE > 1965.4468994140625) and (CASH_ADVANCE_TRX > 3.5)\n\n[0.7808219178082192] (PURCHASES_FREQUENCY <= 0.40833351016044617) and (CASH_ADVANCE_TRX <= 5.5) and (BALANCE <= 2888.3736572265625) and (PRC_FULL_PAYMENT <= 0.5357145071029663) and (CASH_ADVANCE > 2067.3479614257812) and (CREDIT_LIMIT > 4250.0)\n\n[0.8035714285714286] (PURCHASES_FREQUENCY <= 0.40833351016044617) and (CASH_ADVANCE_TRX <= 5.5) and (BALANCE > 2888.3736572265625) and (CASH_ADVANCE_FREQUENCY <= 0.08712099865078926) and (PAYMENTS > 1238.6702270507812)\n\n[0.8571428571428571] (PURCHASES_FREQUENCY <= 0.40833351016044617) and (CASH_ADVANCE_TRX <= 5.5) and (BALANCE > 2888.3736572265625) and (CASH_ADVANCE_FREQUENCY > 0.08712099865078926)\n\n[0.9833333333333333] (PURCHASES_FREQUENCY <= 0.40833351016044617) and (CASH_ADVANCE_TRX > 5.5) and (CASH_ADVANCE <= 6043.52734375) and (CREDIT_LIMIT <= 1750.0) and (CASH_ADVANCE_FREQUENCY > 0.4365074932575226)\n\n[0.5833333333333334] (PURCHASES_FREQUENCY <= 0.40833351016044617) and (CASH_ADVANCE_TRX > 5.5) and (CASH_ADVANCE <= 6043.52734375) and (CREDIT_LIMIT > 1750.0) and (BALANCE <= 5575.639404296875) and (CASH_ADVANCE_TRX <= 22.5) and (BALANCE <= 670.0019226074219)\n\n[0.918918918918919] (PURCHASES_FREQUENCY <= 0.40833351016044617) and (CASH_ADVANCE_TRX > 5.5) and (CASH_ADVANCE <= 6043.52734375) and (CREDIT_LIMIT > 1750.0) and (BALANCE <= 5575.639404296875) and (CASH_ADVANCE_TRX <= 22.5) and (BALANCE > 670.0019226074219)\n\n[0.52] (PURCHASES_FREQUENCY <= 0.40833351016044617) and (CASH_ADVANCE_TRX > 5.5) and (CASH_ADVANCE <= 6043.52734375) and (CREDIT_LIMIT > 1750.0) and (BALANCE <= 5575.639404296875) and (CASH_ADVANCE_TRX > 22.5)\n\n[0.6774193548387096] (PURCHASES_FREQUENCY <= 0.40833351016044617) and (CASH_ADVANCE_TRX > 5.5) and (CASH_ADVANCE <= 6043.52734375) and (CREDIT_LIMIT > 1750.0) and (BALANCE > 5575.639404296875) and (PAYMENTS <= 1925.768310546875)\n\n[0.6938775510204082] (PURCHASES_FREQUENCY > 0.40833351016044617) and (PURCHASES <= 1410.7749633789062) and (CASH_ADVANCE > 1172.3317260742188) and (CASH_ADVANCE <= 4500.915283203125) and (CASH_ADVANCE_FREQUENCY > 0.23611100018024445)\n\n
3           2            1231                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         [0.8448275862068966] (PURCHASES_FREQUENCY > 0.40833351016044617) and (PURCHASES <= 1410.7749633789062) and (CASH_ADVANCE <= 1172.3317260742188) and (PURCHASES_FREQUENCY > 0.5773810148239136) and (BALANCE <= 3233.2625732421875) and (PURCHASES > 1066.5900268554688) and (PURCHASES_TRX > 20.5) and (CREDIT_LIMIT > 5850.0)\n\n[0.4810126582278481] (PURCHASES_FREQUENCY > 0.40833351016044617) and (PURCHASES <= 1410.7749633789062) and (CASH_ADVANCE <= 1172.3317260742188) and (PURCHASES_FREQUENCY > 0.5773810148239136) and (BALANCE > 3233.2625732421875)\n\n[0.49056603773584906] (PURCHASES_FREQUENCY > 0.40833351016044617) and (PURCHASES > 1410.7749633789062) and (PURCHASES <= 5097.239990234375) and (PURCHASES_FREQUENCY <= 0.6833334863185883) and (BALANCE <= 2309.7713623046875) and (PURCHASES > 2279.52001953125)\n\n[0.35714285714285715] (PURCHASES_FREQUENCY > 0.40833351016044617) and (PURCHASES > 1410.7749633789062) and (PURCHASES <= 5097.239990234375) and (PURCHASES_FREQUENCY <= 0.6833334863185883) and (BALANCE > 2309.7713623046875)\n\n[0.6865671641791045] (PURCHASES_FREQUENCY > 0.40833351016044617) and (PURCHASES > 1410.7749633789062) and (PURCHASES <= 5097.239990234375) and (PURCHASES_FREQUENCY > 0.6833334863185883) and (CASH_ADVANCE_TRX <= 12.5) and (PURCHASES_TRX <= 71.5) and (PURCHASES <= 1901.2849731445312) and (PURCHASES_TRX <= 24.5) and (CREDIT_LIMIT > 4050.0)\n\n[0.8402366863905325] (PURCHASES_FREQUENCY > 0.40833351016044617) and (PURCHASES > 1410.7749633789062) and (PURCHASES <= 5097.239990234375) and (PURCHASES_FREQUENCY > 0.6833334863185883) and (CASH_ADVANCE_TRX <= 12.5) and (PURCHASES_TRX <= 71.5) and (PURCHASES <= 1901.2849731445312) and (PURCHASES_TRX > 24.5)\n\n[0.923943661971831] (PURCHASES_FREQUENCY > 0.40833351016044617) and (PURCHASES > 1410.7749633789062) and (PURCHASES <= 5097.239990234375) and (PURCHASES_FREQUENCY > 0.6833334863185883) and (CASH_ADVANCE_TRX <= 12.5) and (PURCHASES_TRX <= 71.5) and (PURCHASES > 1901.2849731445312)\n\n[0.78125] (PURCHASES_FREQUENCY > 0.40833351016044617) and (PURCHASES > 1410.7749633789062) and (PURCHASES <= 5097.239990234375) and (PURCHASES_FREQUENCY > 0.6833334863185883) and (CASH_ADVANCE_TRX <= 12.5) and (PURCHASES_TRX > 71.5) and (PURCHASES <= 3506.1199951171875)\n\n
4           1             397                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [0.8135593220338984] (PURCHASES_FREQUENCY <= 0.40833351016044617) and (CASH_ADVANCE_TRX > 5.5) and (CASH_ADVANCE <= 6043.52734375) and (CREDIT_LIMIT > 1750.0) and (BALANCE > 5575.639404296875) and (PAYMENTS > 1925.768310546875)\n\n[0.7972972972972973] (PURCHASES_FREQUENCY <= 0.40833351016044617) and (CASH_ADVANCE_TRX > 5.5) and (CASH_ADVANCE > 6043.52734375)\n\n[0.6629213483146067] (PURCHASES_FREQUENCY > 0.40833351016044617) and (PURCHASES <= 1410.7749633789062) and (CASH_ADVANCE > 1172.3317260742188) and (CASH_ADVANCE > 4500.915283203125)\n\n[0.6779661016949152] (PURCHASES_FREQUENCY > 0.40833351016044617) and (PURCHASES > 1410.7749633789062) and (PURCHASES <= 5097.239990234375) and (PURCHASES_FREQUENCY > 0.6833334863185883) and (CASH_ADVANCE_TRX > 12.5)\n\n
5           3             309                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [0.8] (PURCHASES_FREQUENCY > 0.40833351016044617) and (PURCHASES > 1410.7749633789062) and (PURCHASES <= 5097.239990234375) and (PURCHASES_FREQUENCY > 0.6833334863185883) and (CASH_ADVANCE_TRX <= 12.5) and (PURCHASES_TRX > 71.5) and (PURCHASES > 3506.1199951171875)\n\n[0.6585365853658537] (PURCHASES_FREQUENCY > 0.40833351016044617) and (PURCHASES > 1410.7749633789062) and (PURCHASES > 5097.239990234375) and (PAYMENTS <= 11673.01171875) and (PURCHASES_TRX <= 60.5)\n\n[0.9857142857142858] (PURCHASES_FREQUENCY > 0.40833351016044617) and (PURCHASES > 1410.7749633789062) and (PURCHASES > 5097.239990234375) and (PAYMENTS <= 11673.01171875) and (PURCHASES_TRX > 60.5)\n\n
6           6              30                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      [0.52] (PURCHASES_FREQUENCY > 0.40833351016044617) and (PURCHASES > 1410.7749633789062) and (PURCHASES > 5097.239990234375) and (PAYMENTS > 11673.01171875)\n\n

'''
