import torch
from torch_geometric.data import Data
import networkx as nx

def convert_nx_to_pyg(nx_graph: nx.DiGraph, normalize_scores: bool=False) -> Data:
    # Get node indices
    # Get node features
    one_hot = {"source": [1, 0, 0], "terminal": [0, 1, 0], "regular": [0, 0, 1]}
    x = []
    node_indices = {}
    for i, node in enumerate( nx_graph.nodes()):
        node_indices[node] = i
        # If you have node features, extract them here
        # This is just a placeholder; replace with your actual node features
        node_data = nx_graph.nodes[node]
        if node_data:
            # Example: Extract 'feature' attribute if it exists
            if 'profile' in node_data:
                x.append(one_hot[node_data['profile']])
            else:
                raise Exception("No tiene caracteristica específica")
        else:
            raise Exception("No tiene features")

    # Get edge indices (2 x num_edges)
    edge_index = []
    edge_attr = []
    edge_criticality_scores = []

    for u, v, data in nx_graph.edges(data=True):
        edge_index.append([node_indices[u], node_indices[v]])
        # If you have edge attributes, extract them here
        if data:
            edge_attr.append(data["weight"])
            #edge_criticality_scores.append(0)
            edge_criticality_scores.append(data["criticality_score"])

        else:
            raise Exception("No hay data en el enlace")

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(x, dtype=torch.float)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    edge_y = torch.tensor(edge_criticality_scores, dtype=torch.float) # ground of truth
    if normalize_scores:
        edge_y = (edge_y - edge_y.min())/(edge_y.max() - edge_y.min())
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_y = edge_y)

    return data
