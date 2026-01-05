import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt 
from typing import Optional
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch


def categorical_palette(n: int, cmap_name: str = "hsv"):
    """
    Generate a categorical color palette with `n` visually distinct colors.

    The palette is created by uniformly sampling a continuous Matplotlib
    colormap. This approach is useful when the number of categories is large
    (e.g., >20), where standard categorical colormaps such as ``tab10`` or
    ``tab20`` are insufficient.

    Parameters
    ----------
    n : int
        Number of distinct colors to generate.
    cmap_name : str, optional
        Name of the Matplotlib colormap to sample from. Common choices include
        ``"hsv"``, ``"nipy_spectral"``, or ``"turbo"``. Default is ``"hsv"``.

    Returns
    -------
    list of tuple
        List of RGBA color tuples, each with values in the range [0, 1],
        suitable for use in Matplotlib plotting functions.
    """
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i / n) for i in range(n)]

def plot_simple_graph(G: nx.DiGraph, sources: list, terminals: list, title="Transport Network"):
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

def plot_graph(gdf: gpd.GeoDataFrame, 
               G: nx.DiGraph, 
               node_color: Optional[str | list[str]]=None, 
               source_target: Optional[tuple[list,list]]=None,
               ax: Optional[plt.Axes] = None):    
    """
    Plot a street/network graph over a GeoDataFrame boundary/background.

    This function draws the geometries from `gdf` (typically a polygon/area of
    interest) and overlays the graph `G` using OSMnx's plotting utilities.
    If `source_target` is provided, source nodes are highlighted in green and
    target nodes in red (with larger node size).

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to plot as a base layer (e.g., area boundary). It is drawn
        with transparent facecolor and grey edges.
    G : networkx.MultiDiGraph or networkx.Graph
        Graph to plot (commonly an OSMnx graph).
    node_color : str or list-like, optional
        Node color(s) passed to `ox.plot.plot_graph`. If `source_target` is
        provided, this is overridden by the source/target coloring scheme.
        Default is "blue" when both `node_color` and `source_target` are None.
    source_target : tuple, optional
        Tuple of the form `(sources, targets)` where each element is an iterable
        of node identifiers. Source nodes are colored green and target nodes red.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. If None, a new figure and axes are created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the rendered plot.
    """
    if isinstance(G, nx.DiGraph) and not isinstance(G, nx.MultiDiGraph):
        G = nx.MultiDiGraph(G)

    node_size = 6
    if node_color is None and source_target is None:
        node_color = "blue"
    elif source_target is not None:
        node_size, node_color = _set_node_info_target_source(G, source_target)


    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10), facecolor="white")
        ax.set_facecolor("white")

    gdf.plot(
    ax=ax,
    facecolor="none",
    edgecolor="lightgrey",
    linewidth=0.6,
    )

    ox.plot.plot_graph(
        G,
        ax=ax,
        show=False,
        close=False,
        bgcolor="white",
        node_color=node_color,
        edge_color="black",
        node_size=node_size,
        edge_linewidth=0.5,
        )
        

    return ax

def _set_node_info_target_source(G: nx.Graph, 
                                 source_target: tuple[list,list]):
    """
    Build node sizes and colors highlighting source and target nodes.

    Nodes are assigned default color "blue" and size 6. Nodes present in the
    `sources` iterable are colored "green" and enlarged; nodes present in the
    `targets` iterable are colored "red" and enlarged.

    Parameters
    ----------
    G : networkx.Graph
        Graph whose nodes will be styled.
    source_target : tuple
        Tuple `(sources, targets)` where `sources` and `targets` are iterables
        of node identifiers.

    Returns
    -------
    tuple
        A tuple `(node_sizes, node_colors)` where:
        - node_sizes : list of int
            Size per node in the order of `list(G.nodes())`.
        - node_colors : list of str
            Color per node in the order of `list(G.nodes())`.
    """
    nodes = list(G.nodes())

    node_colors = ["blue"] * len(nodes)
    node_sizes = [6] * len(nodes)

    source, target = source_target
    source_set = set(source)
    target_set = set(target)

    for i, n in enumerate(nodes):
        if n in source_set:
            node_colors[i] = "green"
            node_sizes[i]  = 60
        if n in target_set:
            node_colors[i] = "red"
            node_sizes[i] = 60
    return node_sizes, node_colors

def _set_graph_title(ax: plt.Axes, 
                     title: str, 
                     G: nx.Graph):
    """
    Set a two-line title on a Matplotlib Axes with graph metadata.

    The first line displays a descriptive title. The second line reports
    basic graph statistics, including the number of nodes and edges.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which the title will be set.
    title : str
        Main title describing the graph.
    G : networkx.Graph
        Graph whose size statistics (nodes and edges) are displayed.

    Returns
    -------
    None
        This function modifies the Axes in place.
    """
    ax.set_title(
        f"{title}" "\n"
        f"Nodes: {G.number_of_nodes()} - Edges: {G.number_of_edges()}"
    )


def plot_graphs_side_by_side(gdf: gpd.GeoDataFrame, 
                             G_left: nx.Graph, 
                             G_right: nx.Graph,
                             node_color_left: Optional[list] = None,
                             node_color_right: Optional[list] = None,
                             source_target_left: Optional[tuple[list,list]] = None,
                             source_target_right: Optional[tuple[list,list]] = None):
    """
    Plot two graphs side-by-side over the same GeoDataFrame background.

    This helper creates a 1x2 subplot layout and calls `plot_graph` for each
    graph, reusing the same base `gdf`. It is useful for visual comparisons
    (e.g., before/after, two routing solutions, etc.).

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to plot as a base layer on both subplots.
    G_left : networkx.Graph
        Graph to plot on the left subplot.
    G_right : networkx.Graph
        Graph to plot on the right subplot.
    node_color_left : str or list-like, optional
        Node color(s) for the left graph (ignored if `source_target_left` is set).
    node_color_right : str or list-like, optional
        Node color(s) for the right graph (ignored if `source_target_right` is set).
    source_target_left : tuple, optional
        `(sources, targets)` for the left plot; highlights sources in green and
        targets in red.
    source_target_right : tuple, optional
        `(sources, targets)` for the right plot; highlights sources in green and
        targets in red.

    Returns
    -------
    None
        This function displays the plot via `plt.show()` and does not return axes.
    """
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(1, 3, width_ratios=[1, 0.03, 1], wspace=0)

    ax_l = fig.add_subplot(gs[0])
    ax_sep = fig.add_subplot(gs[1])
    ax_r = fig.add_subplot(gs[2])
    
    ax_sep.axis("off")
    ax_sep.plot([0.5, 0.5], [0, 1], color="lightgray", linewidth=1.5)

    ax_l.set_facecolor("white")
    ax_r.set_facecolor("white")

    plot_graph(gdf, G_left, node_color=node_color_left,
        source_target=source_target_left, ax=ax_l)

    plot_graph(gdf, G_right, node_color=node_color_right, source_target=source_target_right, ax=ax_r)

    _set_graph_title(ax_l, 
                     "Original transport network with cluster groups", G_left)
    _set_graph_title(ax_r, 
                     "Simplified network", G_right)

    plt.show()


def visualize_flow_network(G: nx.DiGraph, 
                           sources: list, 
                           terminals: list, 
                           title="Transport Network"):
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

    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Sources"),
        Patch(facecolor="lightcoral", edgecolor="black", label="Terminals"),
        Patch(facecolor="lightblue", edgecolor="black", label="Intermediate nodes"),
    ]
    plt.legend(handles=legend_elements, loc="upper left", fontsize=12)
    plt.tight_layout()
