import matplotlib.pyplot as plt
import networkx as nx


def visualize_simple_graph(G: nx.DiGraph, sources: list, terminals: list, title="Transport Network"):
    """
    Visualize a directed graph with sources and terminals highlighted.

    Parameters
    ----------
    G : nx.DiGraph
        The graph to visualize
    sources : list
        List of source nodes
    terminals : list
        List of terminal nodes
    title : str
        Title for the plot
    pos : dict or None
        Node positions. If None, will use hierarchical layout
    """
    plt.figure(figsize=(10, 5))

    # Use hierarchical layout if position not provided
    # Create layers based on shortest path from sources
    layers = {}
    for node in G.nodes():
        if node in sources:
            layers[node] = 0
        elif node in terminals:
            # Put terminals at the end
            layers[node] = 3
        else:
            # Find minimum distance from any source
            min_dist = float("inf")
            for source in sources:
                try:
                    dist = nx.shortest_path_length(G, source, node)
                    min_dist = min(min_dist, dist)
                except nx.NetworkXNoPath:
                    pass
            layers[node] = min_dist if min_dist != float("inf") else 2

    # Create positions based on layers
    pos = {}
    layer_counts = {}
    for node, layer in layers.items():
        if layer not in layer_counts:
            layer_counts[layer] = 0
        layer_counts[layer] += 1

    layer_positions = {layer: 0 for layer in layer_counts}

    for node in sorted(G.nodes(), key=lambda n: (layers[n], n)):
        layer = layers[node]
        total_in_layer = layer_counts[layer]
        y_position = layer_positions[layer] - (total_in_layer - 1) / 2
        pos[node] = (layer * 3, y_position * 2)
        layer_positions[layer] += 1

    # Draw nodes with different colors for sources, terminals, and others
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node in sources:
            node_colors.append("lightgreen")
            node_sizes.append(1500)
        elif node in terminals:
            node_colors.append("lightcoral")
            node_sizes.append(1500)
        else:
            node_colors.append("lightblue")
            node_sizes.append(1200)

    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, edgecolors="black", linewidths=2
    )
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight="bold")

    # Draw edges with curved connections to avoid overlap
    for edge in G.edges():
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[edge],
            edge_color="gray",
            arrows=True,
            arrowsize=25,
            arrowstyle="->",
            width=2.5,
            connectionstyle="arc3,rad=0.2",
            alpha=0.7,
            node_size=node_sizes,
        )

    # Draw edge labels (weight and capacity)
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        weight = data.get("weight", "")
        capacity = data.get("capacity", "")
        edge_labels[(u, v)] = f"w:{weight}, c:{capacity}"

    # Position edge labels with offset to avoid overlap
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels, font_size=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
    )

    plt.title(title, fontsize=18, fontweight="bold", pad=20)
    plt.axis("off")

    # Create legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Sources"),
        Patch(facecolor="lightcoral", edgecolor="black", label="Terminals"),
        Patch(facecolor="lightblue", edgecolor="black", label="Intermediate nodes"),
    ]
    plt.legend(handles=legend_elements, loc="upper left", fontsize=12)
    plt.tight_layout()
