import networkx as nx

def get_sources_targets(G: nx.DiGraph, 
                        n_sources: int, 
                        n_targets: int) -> tuple[list, list]:
    """
    Select source and target nodes based on degree statistics.

    Source nodes are defined as the nodes with the highest out-degree
    (i.e., nodes that send flow to many others).  
    Target nodes are defined as the nodes with the lowest in-degree
    (i.e., nodes that receive little or no flow).

    Parameters
    ----------
    G : nx.DiGraph
        Directed graph from which sources and targets are selected.
    n_sources : int
        Number of source nodes to select (highest out-degree).
    n_targets : int
        Number of target nodes to select (lowest in-degree).

    Returns
    -------
    tuple[list, list]
        A tuple `(sources, targets)` where:
        - `sources` is a list of node identifiers with highest out-degree.
        - `targets` is a list of node identifiers with lowest in-degree.
    """
    sources = sorted(dict(G.out_degree).items(), 
                     key = lambda x: x[1], reverse=True)[0:n_sources]
    sources = [i[0] for i in sources]

    targets = sorted(dict(G.in_degree).items(), 
                    key = lambda x: x[1], reverse=False)[0:n_targets]
    targets = [i[0] for i in targets]
    
    return sources, targets

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