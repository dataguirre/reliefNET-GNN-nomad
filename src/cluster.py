from sklearn.cluster import HDBSCAN
import networkx as nx
import numpy as np 

def cluster_graph(G: nx.Graph, 
                  min_cluster_size: int = 35):
    """
    Cluster graph nodes based on their spatial coordinates (x, y) using HDBSCAN.

    The function extracts each node's projected/geographic coordinates from the
    node attributes (`'x'` and `'y'`), fits an HDBSCAN clustering model, and
    assigns a cluster label to every node. Noise points are labeled as `-1`.

    Parameters
    ----------
    G : networkx.Graph
        Input graph whose nodes contain coordinate attributes:
        - 'x' : float
            Node x-coordinate (e.g., longitude / projected x)
        - 'y' : float
            Node y-coordinate (e.g., latitude / projected y)
    min_cluster_size : int, optional
        Minimum number of nodes required to form a cluster in HDBSCAN.
        Default is 35.

    Returns
    -------
    X : np.ndarray of shape (n_nodes, 2)
        Array of node coordinates in the same order as `G.nodes()`, where each
        row is `[x, y]`.
    hdbscan : HDBSCAN
        Fitted HDBSCAN model instance (configured to store centroid information).
    clusters : np.ndarray of shape (n_nodes,)
        Cluster label per node aligned with `X` and `G.nodes()`. Noise points
        have label `-1`.
    dict_node_cluster : dict
        Mapping `{node_id: cluster_label}` for all nodes in `G`.
    """
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

    