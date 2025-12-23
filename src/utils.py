import itertools

import networkx as nx
from networkx.algorithms.connectivity import build_auxiliary_edge_connectivity, local_edge_connectivity
from networkx.algorithms.flow import build_residual_network


def global_efficiency(G: nx.DiGraph, sources: list, terminals: list) -> float:
    """
    Calculate the global efficiency between source and terminal nodes.

    Global efficiency is computed as the average of the inverse shortest path
    lengths between all source-terminal pairs.

    Parameters
    ----------
    G : nx.DiGraph
        A directed graph with weighted edges.
    sources : list
        List of source node identifiers.
    terminals : list
        List of terminal node identifiers.

    Returns
    -------
    float
        The global efficiency value, normalized by the number of source-terminal pairs.
    """
    D = dict(nx.all_pairs_dijkstra_path_length(G))
    sum_distances = 0
    for s, t in itertools.product(sources, terminals):
        sum_distances += 1 / D[s][t]
    return (1 / (len(sources) * len(terminals))) * sum_distances


def number_independent_paths(G: nx.DiGraph, sources: list, terminals: list) -> float:
    """
    Calculate the average number of edge-independent paths between sources and terminals.

    This function computes the local edge connectivity for all source-terminal pairs
    and returns the normalized sum.

    Parameters
    ----------
    G : nx.DiGraph
        A directed graph.
    sources : list
        List of source node identifiers.
    terminals : list
        List of terminal node identifiers.

    Returns
    -------
    float
        The average number of independent paths, normalized by the number of terminals.
    """
    H = build_auxiliary_edge_connectivity(G)
    R = build_residual_network(H, "capacity")
    k = 0
    for s, t in itertools.product(sources, terminals):
        k += local_edge_connectivity(G, s, t, auxiliary=H, residual=R)
    return (1 / (len(terminals))) * k


def max_flow(G: nx.DiGraph, sources: list, terminals: list) -> float:
    """
    Calculate the average maximum flow between source and terminal nodes.

    This function computes the maximum flow value for all source-terminal pairs
    and returns the normalized average.

    Parameters
    ----------
    G : nx.DiGraph
        A directed graph with edges containing a 'capacity' attribute.
    sources : list
        List of source node identifiers.
    terminals : list
        List of terminal node identifiers.

    Returns
    -------
    float
        The average maximum flow value, normalized by the number of source-terminal pairs.

    Notes
    -----
    Graph edges must have a 'capacity' attribute for this function to work correctly.
    """
    flow_value = 0
    for s, t in itertools.product(sources, terminals):
        flow_value += nx.maximum_flow_value(G, s, t)  # Be aware "capacity" key must be in edge attributes
    return (1 / (len(sources) * len(terminals))) * flow_value
