import itertools

import networkx as nx
from networkx.algorithms.connectivity import build_auxiliary_edge_connectivity, local_edge_connectivity
from networkx.algorithms.flow import build_residual_network


def global_efficiency(G: nx.DiGraph, sources: list, terminals: list) -> float:
    D = dict(nx.all_pairs_dijkstra_path_length(G))
    N = len(sources) + len(terminals)
    sum_distances = 0
    for s, t in itertools.product(sources, terminals):
        sum_distances += 1 / D[s][t]
    return (1 / (len(sources) * len(terminals))) * sum_distances


def number_independent_paths(G: nx.DiGraph, sources: list, terminals: list) -> float:
    H = build_auxiliary_edge_connectivity(G)
    R = build_residual_network(H, "capacity")
    k = 0
    for s, t in itertools.product(sources, terminals):
        k += local_edge_connectivity(G, s, t, auxiliary=H, residual=R)
    return (1 / (len(terminals))) * k


def max_flow(G: nx.DiGraph, sources: list, terminals: list) -> float:
    # Los enlaces deben tener como llave "capacity"
    flow_value = 0
    for s, t in itertools.product(sources, terminals):
        flow_value += nx.maximum_flow_value(G, s, t)
    return (1 / (len(sources) * len(terminals))) * flow_value
