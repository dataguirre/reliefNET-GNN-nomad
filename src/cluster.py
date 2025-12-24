from sklearn.cluster import HDBSCAN
import numpy as np 

def cluster_graph(G, min_cluster_size = 35):
    nodes = G.nodes(data = True)

    X = []
    for id, info in nodes:
        X.append([info['x'], info['y']])
    X = np.array(X)

    hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', 
                      store_centers='centroid', copy=False )
    clusters = hdbscan.fit_predict(X)
    dict_node_cluster = {i: int(c) for i,c in zip(G.nodes, clusters)}

    return X, hdbscan, clusters, dict_node_cluster

    