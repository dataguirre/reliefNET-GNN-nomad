import networkx as nx

def get_sources_targets(G: nx.DiGraph, 
                        n_sources: int, 
                        n_targets: int) -> nx.DiGraph:
    """
    Label nodes in a directed graph as sources, targets, or regular
    based on degree statistics.

    Source nodes are defined as the nodes with the highest out-degree
    (i.e., nodes that send flow to many others).
    Target nodes are defined as the nodes with the lowest in-degree
    (i.e., nodes that receive little or no flow).

    The function assigns a node attribute ``'profile'`` with one of
    the following values:
    - ``'source'``  : node selected among the top ``n_sources`` by out-degree
    - ``'target'``  : node selected among the bottom ``n_targets`` by in-degree
    - ``'regular'`` : all other nodes

    Parameters
    ----------
    G : nx.DiGraph
        Directed graph whose nodes will be labeled.
    n_sources : int
        Number of source nodes to label (highest out-degree).
    n_targets : int
        Number of target nodes to label (lowest in-degree).

    Returns
    -------
    nx.DiGraph
        The same graph ``G`` with an added node attribute ``'profile'``.
        The graph is modified in place.
    """
    sources = sorted(dict(G.out_degree).items(), 
                     key = lambda x: x[1], reverse=True)[0:n_sources]
    sources = [i[0] for i in sources]

    targets = sorted(dict(G.in_degree).items(), 
                    key = lambda x: x[1], reverse=False)[0:n_targets]
    targets = [i[0] for i in targets]

    for node, data in G.nodes(data=True):
        if node in sources:
            data['profile'] = 'source'
        elif node in targets:
            data['profile'] = 'target'
        else:
            data['profile'] = 'regular'
    
    return G

def create_capacities_from_length(G: nx.DiGraph) -> None:
    """
    Initialize edge capacities using edge lengths.

    For every edge in the graph, this function creates (or overwrites)
    the `capacity` attribute and sets it equal to the edge's `length`.
    This is useful as a simple baseline when capacities are not available
    and distance-based capacity is acceptable.

    Parameters
    ----------
    G : nx.DiGraph
        Directed graph whose edges contain a `length` attribute.

    Returns
    -------
    None
        Modifies `G` in-place by adding a `capacity` attribute to each edge.
    """
    for e in G.edges(data=True):
        e[2]['capacity'] = e[2]['length']