import concurrent.futures
import itertools
import math
import random
from typing import Callable, Generator, Union

import networkx as nx
from tqdm import tqdm


def get_link_criticality_scores(
    obj: Union[nx.DiGraph, str],
    index_accesibility,
    max_links_in_disruption=0.4,
    max_disruption_scenarios=1_000_000,
    workers=False,
    show_progress=True,
) -> dict:
    """Process a single network - this will be called in parallel"""
    G = obj if isinstance(obj, nx.Graph) else nx.read_gml(obj)
    disruptions = _generate_disruption_scenarios(
        list(G.edges), max_links_in_disruption=max_links_in_disruption, max_scenarios=max_disruption_scenarios
    )
    total_disruption_scenarios = min(
        _get_total_disruption_scenarios(len(G.edges), max_links_in_disruption), max_disruption_scenarios
    )
    print(f"Testing with {total_disruption_scenarios} disruption scenarios")

    fn_criticality_score = _criticality_score
    if workers or workers is None:
        fn_criticality_score = _parallel_criticality_score

    # Call the  ranking function with inner parallelism
    link_scores = fn_criticality_score(
        G,
        disruptions,
        index_accesibility,
        max_workers=workers,
        max_disruption_scenarios=max_disruption_scenarios,
        max_links_in_disruption=max_links_in_disruption,
        show_progress=show_progress,
        total_disruption_scenarios=total_disruption_scenarios,
    )

    # Return results
    result = {
        "network_path": str(obj),
        "link_scores": list(link_scores.items()),
        "disruption_scenarios": total_disruption_scenarios,
    }
    return result


def _criticality_score(
    G: nx.DiGraph, disruptions: Generator, accesibility_index: Callable[[nx.DiGraph, list, list], float], **kwargs
) -> list[tuple]:
    """rank links based on accesibility index

    Parameters
    ----------
    G : nx.DiGraph
        Graph with sources and terminals and weights as time
    disruptions : list[tuple[tuple]]
        list of disruptions. Each disruption consist of 1 or more
        edges
    accesibility_index : Callable[[nx.DiGraph, list, list], float]
        function that evaluate the graph with/out disruptions

    Returns
    -------
    list[tuple]
        list of edges sorted with most to lowest criticality to the
        network

    """
    assert _all_edges_have_attr(G, "weight")

    sources, terminals = [], []
    for node, profile in nx.get_node_attributes(G, "profile").items():
        if profile == "source":
            sources.append(node)
        elif profile == "terminal":
            terminals.append(node)
    if not sources or not terminals:
        raise Exception("There must be at least 1 source and 1 terminal")

    P_0 = accesibility_index(G, sources, terminals)  # initial performance of network
    link_performance = {}
    # breakpoint()
    for disruption in tqdm(disruptions, total=kwargs["total_disruption_scenarios"]):
        _, delta = _evaluate_disruption(disruption, G, sources, terminals, accesibility_index, P_0)
        # if delta:
        for edge in disruption:
            # normalize the delta by the size of the disruption:
            link_performance[edge] = link_performance.get(edge, 0) + delta * 1 / len(disruption)

    return link_performance


def _parallel_criticality_score(
    G, disruptions, accessibility_index, max_workers=None, batch_size=100, show_progress=True, **kwargs
):
    """Parallelized version of rank_links with batching for better performance"""

    # Identify sources and terminals
    sources, terminals = [], []
    for node, profile in nx.get_node_attributes(G, "profile").items():
        if profile == "source":
            sources.append(node)
        elif profile == "terminal":
            terminals.append(node)
    if not sources or not terminals:
        raise Exception("There must be at least 1 source and 1 terminal")

    # Calculate initial network performance
    P_0 = accessibility_index(G, sources, terminals)

    # Initialize result container
    link_performance = {}

    # Process disruptions in batches
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        batch = []

        # Submit batches while preserving generator behavior
        for disruption in disruptions:
            batch.append(disruption)
            if len(batch) >= batch_size:
                batch_data = (batch, G, sources, terminals, accessibility_index, P_0)
                futures.append(executor.submit(_process_batch, batch_data))
                batch = []

        # Submit any remaining disruptions
        if batch:
            batch_data = (batch, G, sources, terminals, accessibility_index, P_0)
            futures.append(executor.submit(_process_batch, batch_data))

        # Process results as they complete
        if show_progress:
            total_futures = len(futures)
            iterator = tqdm(concurrent.futures.as_completed(futures), total=total_futures, desc="Processing batches")
        else:
            iterator = concurrent.futures.as_completed(futures)
        for future in iterator:
            batch_results = future.result()
            for disruption, delta in batch_results:
                for edge in disruption:
                    link_performance[edge] = link_performance.get(edge, 0) + delta * (1 / len(disruption))

    # Return results (as a dictionary rather than sorted list to match your original function)
    return link_performance


def _process_batch(batch_data):
    """Process a batch of disruption scenarios

    Parameters
    ----------
    batch_data : tuple
        A tuple containing (batch, G, sources, terminals, accessibility_index, P_0)

    Returns
    -------
    list
        List of (disruption, delta) tuples for each processed disruption
    """
    batch, G, sources, terminals, accessibility_index, P_0 = batch_data
    batch_results = []
    for disruption in batch:
        try:
            G_disrupted = G.copy()
            for edge in disruption:
                G_disrupted.remove_edge(*edge)

            P_disruption = accessibility_index(G_disrupted, sources, terminals)
            delta = P_0 - P_disruption
            batch_results.append((disruption, delta))
        except:
            batch_results.append((disruption, 0))
    return batch_results


def _evaluate_disruption(disruption, G, sources, terminals, accessibility_index, P_0):
    """Evaluate a single disruption scenario"""
    G_disrupted = G.copy()
    for edge in disruption:
        G_disrupted.remove_edge(*edge)
    # try:
    P_disruption = accessibility_index(G_disrupted, sources, terminals)
    delta = P_0 - P_disruption
    # Return the disruption and its impact
    del G_disrupted
    return disruption, delta
    # except:
    #     return disruption, 0


def _all_edges_have_attr(G: nx.Graph, attr: str) -> bool:
    """Check if all edges in the graph have a specific attribute."""
    return all(attr in data for _, _, data in G.edges(data=True))


def _generate_disruption_scenarios(edges: list, max_links_in_disruption: float, max_scenarios: int = 2_000_000):
    """
    Generate disruption scenarios by selecting combinations of edges.

    Args:
        edges: List of edges in the network
        max_links_in_disruption: Percentage of links to consider for disruption (0.0 to 1.0)
        max_scenarios: Maximum number of scenarios to generate

    Returns:
        Generator yielding disruption scenarios
    """
    R = max(1, int(len(edges) * max_links_in_disruption))

    # Calculate total number of combinations for all r values
    total_combinations = 0
    combinations_by_r = {}

    for r in range(1, R + 1):
        try:
            n_combinations = math.comb(len(edges), r)
            combinations_by_r[r] = n_combinations
            total_combinations += n_combinations
        except OverflowError:
            # For extremely large combinations
            combinations_by_r[r] = float("inf")
            total_combinations = float("inf")
            break

    # If total combinations don't exceed max_scenarios, generate all combinations
    if total_combinations <= max_scenarios:
        for r in range(1, R + 1):
            yield from itertools.combinations(edges, r)
        return

    # Always include all single-edge failures first
    yield from itertools.combinations(edges, 1)
    scenarios_generated = combinations_by_r.get(1, len(edges))

    # Distribute remaining scenarios across r = 2 to R
    if scenarios_generated >= max_scenarios:
        return

    remaining_scenarios = max_scenarios - scenarios_generated

    # Allocate remaining scenarios proportionally across r values
    for r in range(2, R + 1):
        if remaining_scenarios <= 0:
            break

        n_combinations = combinations_by_r.get(r, 0)

        if n_combinations <= remaining_scenarios:
            # If we can generate all combinations for this r, do so
            yield from itertools.combinations(edges, r)
            remaining_scenarios -= n_combinations
        else:
            # Otherwise, sample from the combinations
            if n_combinations > 10000:  # For very large combination spaces, use sampling
                edge_indices = list(range(len(edges)))
                sampled_indices_sets = set()  # To avoid duplicates

                while len(sampled_indices_sets) < remaining_scenarios:
                    sampled_indices = tuple(sorted(random.sample(edge_indices, r)))
                    if sampled_indices not in sampled_indices_sets:
                        sampled_indices_sets.add(sampled_indices)
                        yield tuple(edges[i] for i in sampled_indices)
            else:
                # For smaller spaces, generate combinations with a limit
                for i, combo in enumerate(itertools.combinations(edges, r)):
                    if i >= remaining_scenarios:
                        break
                    yield combo

            break  # We've used up all remaining scenarios


def _get_total_disruption_scenarios(n_edges: int, max_links_in_disruption: float) -> int:
    """Get the total disruption scenarios based on number of edges and max_links_in_disruption disruptions"""
    R = int(n_edges * max_links_in_disruption)
    total = sum(math.comb(n_edges, r) for r in range(1, R + 1))
    return total
