from scipy.spatial.distance import cdist
from sklearn.cluster import HDBSCAN
from shapely.geometry import LineString
from shapely.ops import linemerge
import geopandas as gpd
import networkx as nx
import osmnx as ox
import numpy as np

from src.plot import plot_graphs_side_by_side

def build_graph(cities: list[str])->tuple[gpd.GeoDataFrame,nx.MultiDiGraph]:
    """
    Build a GeoDataFrame and a driving network graph for the given places.

    This function geocodes the input place(s) into a GeoDataFrame and downloads
    a drivable street network using OSMnx.

    Parameters
    ----------
    cities : str or list-like
        Place name(s) accepted by OSMnx (e.g., a single city name, or a list of
        place strings).

    Returns
    -------
    tuple
        A tuple `(gdf, G)` where:
        - gdf : gpd.GeoDataFrame
            GeoDataFrame with the geocoded place geometry.
        - G : networkx.MultiDiGraph
            OSMnx street network graph for driving.
    """
    gdf = ox.geocode_to_gdf(cities)
    G = ox.graph.graph_from_place(cities, network_type="drive", simplify=True)

    return gdf, G


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

def _find_closest_point(points: np.array, 
                        centroid: np.array) -> int:
    """
    Find the index of the point closest to a centroid using Euclidean distance.

    Distances are computed with `scipy.spatial.distance.cdist`.

    Parameters
    ----------
    points : array-like of shape (n_samples, n_features)
        Point coordinates (e.g., node embeddings or (x, y) coordinates).
    centroid : array-like of shape (n_features,)
        Centroid coordinates.

    Returns
    -------
    int
        Index of the closest point in `points` to `centroid`.
    """
    D = cdist(points, np.array([centroid]), metric='euclidean')
    closest_id = np.argmin(D)
    return closest_id

def _find_outside_points_connected_to_cluster(dict_node_cluster: dict, 
                                              cluster_id: int, 
                                              G: nx.Graph) -> set:
    """
    Find nodes outside a cluster that are directly connected to the cluster.

    Scans all edges in `G` and returns the set of endpoints that lie outside
    `cluster_id` but share an edge with a node inside `cluster_id`.

    Parameters
    ----------
    dict_node_cluster : dict
        Mapping `node_id -> cluster_id`.
    cluster_id : int
        Cluster label of interest.
    G : networkx.Graph
        Graph whose edges are inspected.

    Returns
    -------
    set
        Set of node identifiers that are not in `cluster_id` but are adjacent to
        at least one node in `cluster_id`.
    """
    connected_points=set()
    for edge in G.edges():
        id1, id2 = edge
        cluster1, cluster2 = dict_node_cluster[id1], dict_node_cluster[id2]
        if cluster1 == cluster_id and cluster2 != cluster_id:
            connected_points.add(id2)
        elif cluster1 != cluster_id and cluster2 == cluster_id:
            connected_points.add(id1)
    return connected_points

def _find_shortest_path(G: nx.Graph, 
                        s: int, 
                        t: int) -> list:
    """
    Compute the shortest path between two nodes using edge length as weight.

    Parameters
    ----------
    G : networkx.Graph
        Graph containing nodes `s` and `t`. Edge attribute `length` is used
        as the weight.
    s : hashable
        Source node identifier.
    t : hashable
        Target node identifier.

    Returns
    -------
    list
        Sequence of node identifiers representing the shortest path from `s` to `t`.
    """
    return nx.shortest_path(G, source=s, target=t, weight="length")

def _create_missing_geometry(G: nx.Graph,
                             n1: int,
                             n2: int) -> LineString:
    """
    Create a straight LineString between two nodes when edge geometry is missing.

    Coordinates are taken from node attributes `x` and `y`.

    Parameters
    ----------
    n1 : hashable
        First node identifier.
    n2 : hashable
        Second node identifier.
    G : networkx.Graph
        Graph containing node coordinate attributes `x` and `y`.

    Returns
    -------
    shapely.geometry.LineString
        LineString connecting `(x1, y1)` to `(x2, y2)`.
    """
    x1, y1 = G.nodes[n1]['x'], G.nodes[n1]['y']
    x2, y2 = G.nodes[n2]['x'], G.nodes[n2]['y']
    p1 = (x1, y1)
    p2 = (x2, y2)
    return LineString([p1, p2])

def _unify_path(G: nx.MultiDiGraph, 
                path: list) -> dict:
    """
    Aggregate a multi-edge path into a single edge description.

    The function sums the `length` of each edge along the path and merges
    their geometries into a single LineString/MultiLineString using `linemerge`.
    If an edge has no `geometry`, a straight segment is created between the
    corresponding nodes.

    Parameters
    ----------
    G : networkx.Graph
        Graph where edges contain at least `length`, and optionally `geometry`.
        For MultiDiGraphs, this uses the first key `[0]` for each (u, v) pair.
    path : list
        Sequence of node identifiers describing the path.

    Returns
    -------
    dict
        Dictionary with unified edge information:
        - 'geometry' : shapely geometry representing the merged path
        - 'length' : float total length of the path
    """
    lenght = 0
    geometry = None 
    for n1, n2 in zip(path, path[1:]):
        edge = G.get_edge_data(n1,n2)[0]
        lenght += edge['length']
        
        seg = edge.get("geometry") or _create_missing_geometry(G, n1, n2)

        if geometry is None:
            geometry = seg
        else:
            geometry = linemerge([geometry, seg])
    new_information = {
        'geometry': geometry,
        'length': lenght
    }
    return new_information

def _delete_nodes_in_cluster(G:nx.Graph, 
                             center: int, 
                             cluster_id: int, 
                             dict_node_cluster: dict) -> None:
    """
    Remove all nodes in a cluster except the chosen center node.

    Parameters
    ----------
    G : networkx.Graph
        Graph to modify in-place.
    center : hashable
        Node identifier that will be kept.
    cluster_id : int
        Cluster label whose nodes will be pruned.
    dict_node_cluster : dict
        Mapping `node_id -> cluster_id`.

    Returns
    -------
    None
        Modifies `G` in-place.
    """
    nodes = list(G.nodes())
    for n in nodes:
        if dict_node_cluster[n] == cluster_id and n != center:
            G.remove_node(n)    

def _solve_multiple_edges_graph(G: nx.MultiDiGraph) -> None:
    """
    Resolve parallel edges by splitting extra edges into intermediate nodes.

    For each pair (u, v) that has multiple edge keys, all edges except the first
    are replaced by a two-edge path u -> new_node -> v (see `_solve_edge`).

    Parameters
    ----------
    G : networkx.MultiDiGraph
        Graph that may contain multiple edges between the same node pair.

    Returns
    -------
    None
        Modifies `G` in-place.
    """
    for (u,v) in set(G.edges()):
        keys = list(G[u][v].keys())
        for k in keys[1:]:
            _solve_edge(G, u, v, k)

def _solve_edge(G: nx.MultiDiGraph, 
                u: int, 
                v: int, 
                i: int) -> None:
    """
    Replace one parallel edge (u, v, key=i) by routing it through a new node.

    A new node is created (copying attributes from `u`). The selected parallel
    edge is removed and replaced with:
    - u -> new_node with length 0 and a "missing" straight geometry
    - new_node -> v with the original edge geometry/length

    Parameters
    ----------
    G : networkx.MultiDiGraph
        Graph to modify in-place.
    u : hashable
        Edge source node identifier.
    v : hashable
        Edge target node identifier.
    i : int or hashable
        Edge key to resolve for the pair (u, v).

    Returns
    -------
    None
        Modifies `G` in-place.
    """
    id_new_node = max(G.nodes()) + 1
    new_node = G.nodes[u].copy()
    G.add_node(id_new_node, **new_node)

    edge_info = G.get_edge_data(u, v, i)
    new_info_u_new = {
                "geometry": _create_missing_geometry(G, u, id_new_node),
                "length": 0.0,
                #"capacity": edge_info['capacity']
            }
    G.add_edge(u,id_new_node,**new_info_u_new)

    if "geometry" in edge_info:
        geometry = edge_info['geometry']
    else:
        geometry = _create_missing_geometry(G, u, v)
            
    new_info_new_v = {
                "geometry": geometry,
                "length": edge_info['length'],
                #"capacity": edge_info['capacity']
            }
    G.add_edge(id_new_node,v,**new_info_new_v)
    G.remove_edge(u, v, key=i)


def _clean_edges(G: nx.MultiDiGraph) -> None:
    """
    Clean graph edges by removing self-loops and resolving parallel edges.

    Steps:
    1) Remove self-loop edges (including keys).
    2) Convert remaining parallel edges using `_solve_multiple_edges_graph`.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        Graph to clean.

    Returns
    -------
    None
        Modifies `G` in-place.
    """
    self_loops = list(nx.selfloop_edges(G, keys=True))
    G.remove_edges_from(self_loops)

    _solve_multiple_edges_graph(G)

def _update_path_to_other_centroid(G: nx.MultiDiGraph, 
                                G_new: nx.MultiDiGraph,
                                p_point: int, 
                                q_centroid: int, 
                                dict_node_cluster: dict,
                                closest_to_centroids: dict):
    """
    Connect two centroids when an outside point has been removed.

    This function handles the situation where a node `p_point`, originally
    belonging to one cluster, has been removed and then has been identified 
    as an outside point to connect to other centroid.The connectivity between both
    clusters is preserved by directly linking their centroids.

    For each former neighbor `q_point` of `p_point` that belongs to the cluster
    of `q_centroid`, shortest paths are constructed between the corresponding
    centroids through the virtual connection (p_point, q_point). These paths
    are then aggregated into single centroid-to-centroid edges and added to
    the reduced graph `G_new` in both directions.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        Original graph before node removal.
    G_new : networkx.MultiDiGraph
        Reduced graph containing only centroids and aggregated edges.
    p_point : int
        Node that was removed from its original cluster (outside point).
    q_centroid : int
        Centroid of the neighboring cluster that must remain connected.
    dict_node_cluster : dict
        Mapping `node_id -> cluster_id`.
    closest_to_centroids : dict
        Mapping `cluster_id -> centroid_node_id`.

    Returns
    -------
    None
        Modifies `G_new` in-place by adding direct centroid-to-centroid edges.
    """
    cluster_p = dict_node_cluster[p_point]
    cluster_q = dict_node_cluster[q_centroid]
    q_points = []

    for q in list(G.successors(p_point)) + list(G.predecessors(p_point)):
        cluster = dict_node_cluster[q]
        if cluster == cluster_q:
            q_points.append(q)

    for q_point in q_points: 
        p_centroid = closest_to_centroids[cluster_p]
        path_pcentroid_p = _find_shortest_path(G, p_centroid, p_point)
        path_q_qcentroid = _find_shortest_path(G, q_point, q_centroid)
        path_pcentroid_qcentroid = path_pcentroid_p[:-1] + [p_point, q_point] + path_q_qcentroid[1:]
        new_edge_info1 = _unify_path(G = G, path = path_pcentroid_qcentroid)

        G_new.add_edge(p_centroid, q_centroid, **new_edge_info1)

        path_qcentroid_q = _find_shortest_path(G, q_centroid, q_point)
        path_p_pcentroid = _find_shortest_path(G, p_point, p_centroid)
        path_qcentroid_pcentroid = path_qcentroid_q[:-1] + [q_point, p_point] + path_p_pcentroid[1:]
        new_edge_info2 = _unify_path(G = G, path = path_qcentroid_pcentroid)

        G_new.add_edge(q_centroid, p_centroid, **new_edge_info2)

def _unify_clusters(G: nx.MultiDiGraph, 
                    hdbscan: HDBSCAN, 
                    clusters: list[int], 
                    dict_node_cluster: dict, 
                    X: np.array) -> nx.MultiDiGraph:
    """
    Simplify a graph by collapsing each cluster into its centroid-closest node.

    For each cluster label `c` (excluding noise `-1`):
    - Select the node closest to the cluster centroid as the cluster center.
    - Remove all other nodes in that cluster from a copy of the graph.
    - For each outside node directly adjacent to any node in the cluster (in the
      original graph), add two directed edges between center and outside node:
        center -> outside and outside -> center
      Each new edge aggregates the shortest path geometry and length.

    Parameters
    ----------
    G : networkx.Graph
        Original graph used to compute shortest paths and adjacency.
    hdbscan : object
        Clustering object providing `centroids_` indexed by cluster label.
    clusters : array-like
        Cluster label per node (aligned with the ordering used to build
        `dict_node_cluster` and/or `list(G.nodes())` in your pipeline).
    dict_node_cluster : dict
        Mapping `node_id -> cluster_id`.
    X : array-like
        Point matrix aligned with `list(G.nodes())`, used to find the closest
        node to each centroid.

    Returns
    -------
    networkx.Graph
        A new graph where clustered nodes have been collapsed and connections
        rewired through the selected center nodes.
    """
    G_new = G.copy()
    closest_to_centroids = {}
    for c in np.unique(clusters):
        if c == -1: 
            continue

        ctc_id = _find_closest_point(points = X, centroid = hdbscan.centroids_[c])
        closest_to_centroid = list(G.nodes())[ctc_id]
        closest_to_centroids[c] = closest_to_centroid

        _delete_nodes_in_cluster(G = G_new, 
                                cluster_id = c, 
                                dict_node_cluster = dict_node_cluster, 
                                center = closest_to_centroid)
        
        outside_points = _find_outside_points_connected_to_cluster(
            dict_node_cluster = dict_node_cluster,
            cluster_id = c,
            G = G)
        
        for n_outside in outside_points: 
            # Se eliminó
            if n_outside not in G_new.nodes():
                _update_path_to_other_centroid(G = G, G_new = G_new, 
                                                                     p_point = n_outside, q_centroid = closest_to_centroid, 
                                                                     dict_node_cluster = dict_node_cluster, 
                                                                     closest_to_centroids = closest_to_centroids)
            else:
                shortest1 = _find_shortest_path(G = G, s = closest_to_centroid, t = n_outside)
                new_edge_info1 = _unify_path(G = G, path = shortest1)

                G_new.add_edge(closest_to_centroid, n_outside, **new_edge_info1)

                # Camino del de afuera al del centro
                shortest2 = _find_shortest_path(G, s = n_outside, t = closest_to_centroid)
                new_edge_info2 = _unify_path(G = G, path = shortest2)

                G_new.add_edge(n_outside, closest_to_centroid, **new_edge_info2)
            
    return G_new

def simplify_graph(G: nx.MultiDiGraph, 
                   gdf: gpd.GeoDataFrame, 
                   plot:bool = False) -> nx.DiGraph:
    """
    Build, cluster, and simplify a street network by collapsing clustered nodes.

    Workflow:
    1) Download the driving network for `cities` with OSMnx.
    2) Cluster nodes 
    3) Collapse each cluster into the node closest to its centroid and rewire
       connections to outside nodes via shortest paths (geometry + length).
    4) Clean the resulting graph by removing self-loops and resolving parallel edges.
    5) Optionally plot the original and simplified graphs side-by-side.

    Parameters
    ----------
    cities : list[str]
        List of place strings accepted by OSMnx (e.g., ["Cali, Colombia"]).
        These are used to geocode and download the driving network.
    plot : bool, optional
        If True, plot original vs simplified graphs using `plot_graphs_side_by_side`.
        Default is False.

    Returns
    -------
    nx.DiGraph
        Simplified directed graph with unified cluster centers and cleaned edges.
    """
    X, hdbscan, clusters, dict_node_cluster = cluster_graph(G)
    G_simplified = _unify_clusters(G, hdbscan, clusters, dict_node_cluster, X)
    _clean_edges(G_simplified)
    G_simplified = nx.DiGraph(G_simplified)
    
    if plot:
        plot_graphs_side_by_side(gdf = gdf,
                                 G_left = G,
                                 G_right = G_simplified,
                                 node_color_left = clusters)
    
    return G_simplified

# TODO BORRAR
if __name__ == "__main__":
    cities = ['El Carmen, Colombia']
    gdf, G = build_graph(cities)
    G = simplify_graph(G, gdf, plot = True)
