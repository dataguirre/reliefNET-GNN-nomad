import torch
from torch_geometric.nn.models import GAT, GCN
from torchvision.ops import MLP
import lightning as L
from typing import Optional


class GNNCustom(torch.nn.Module):
    def __init__(
        self,
        gnn_type: str,
        num_x_features: int,
        embedding_size: int,
        num_layers_msg: int,
        num_layers_mlp: int,
        dropout=0,
        **kwargs,
    ):
        super(GNNCustom, self).__init__()
        model = GCN if gnn_type == "GCN" else GAT
        self.msg_block = model(
            in_channels=num_x_features,
            hidden_channels=embedding_size,
            num_layers=num_layers_msg,
            dropout=dropout,
            **kwargs,
        )

        mlp_in_channels = 2 * embedding_size + 1

        self.edge_mlp = MLP(
            in_channels=mlp_in_channels,
            hidden_channels=[embedding_size] * num_layers_mlp + [1],
            dropout=dropout,
        )

    def forward(self, x, edge_index, edge_weight):
        # 1. Create node embeddings using the message block
        node_embeddings = self.msg_block(x, edge_index, edge_weight)
        # 2. Create edge embeddings (by concatenating). An edge is node(tail) ---> node(head)
        tails, heads = edge_index
        edge_weight_reshaped = edge_weight.view(-1, 1)
        edge_embeddings = torch.cat([node_embeddings[tails], node_embeddings[heads]], dim=1)
        # add the weight of edge as an extra dimension
        edge_embeddings = torch.cat([edge_embeddings, edge_weight_reshaped], dim=1)
        # 3. pass through the MLP block to return 1 value per edge
        edge_logits = self.edge_mlp(edge_embeddings).squeeze(-1)
        return edge_logits


class LightningGNNCustom(L.LightningModule):

    def __init__(
        self,
        num_x_features: int,
        embedding_size: int,
        num_layers_msg: int,
        num_layers_mlp: int,
        msg_passing: str,
        loss_fn: str,
        dropout: float,
        edge_embedding_operator: str,
        loss_phi_fn: Optional[str] = None,
        lr=1e-3,
        **kwargs,
    ):
        super().__init__()
        assert msg_passing in ["GCN", "GAT"]
        if msg_passing == "GAT":
            assert isinstance(kwargs["v2"], bool)
            assert isinstance(kwargs["heads"], int)
        elif msg_passing == "GCN":
            assert kwargs.get("v2") is None
            assert kwargs.get("heads") is None
        assert edge_embedding_operator in ["hadamard", "mean", "concat"]
        assert loss_fn in ["MSE", "RANK"]
        if loss_phi_fn == "logistic":
            self.loss_phi_fn = lambda z: torch.log(1 + 1 / torch.exp(z))
        elif loss_phi_fn == "exponential":
            self.loss_phi_fn = lambda z: (1 / torch.exp(z))
        elif loss_phi_fn == "sqrt_exp":
            self.loss_phi_fn = lambda z: 1 - torch.sqrt(torch.exp(z) / (1 + torch.exp(z)))
        elif loss_fn == "RANK":
            raise Exception("Loss fn is rank but not a valid loss phi fn")
        else:
            self.loss_phi_fn = None
        self.lr = lr

        self.save_hyperparameters()

        # drop all the None values of the model
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.model = GNNCustom(
            num_x_features=num_x_features,
            embedding_size=embedding_size,
            num_layers_msg=num_layers_msg,
            num_layers_mlp=num_layers_mlp,
            gnn_type=msg_passing,
            dropout=dropout,
            **kwargs,
        )
        self.loss_fn = loss_fn


    def forward(self, batch):
        return self.model(batch.x, batch.edge_index, batch.edge_attr)
