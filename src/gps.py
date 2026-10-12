from typing import Callable, Optional

import torch
from torch import nn
from torch_geometric.nn import (
    GINEConv,
    GINConv,
    GPSConv,
    global_add_pool,
    global_mean_pool,
)

from src.mlp import MLP


class GPSBaseModel(nn.Module):
    """GPS task model with either graph-level or node-level output.

    DIAL is fused into ``X_n`` by ``PEARL_GNN_Model`` before this module is
    called. The separate ``PE`` argument is intentionally not used here, so
    the structural encoding is injected only once.
    """

    def __init__(
        self,
        n_layers: int,
        n_edge_types: int,
        in_dims: int,
        hidden_dims: int,
        create_mlp: Callable[[int, int], MLP],
        pooling: Optional[str] = "add",
        target_dim: int = 1,
        heads: int = 4,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        local_model: str = "gine",
    ) -> None:
        super().__init__()

        if n_layers < 1:
            raise ValueError("GPS requires at least one layer")
        if hidden_dims % heads != 0:
            raise ValueError(
                f"GPS hidden dimension ({hidden_dims}) must be divisible by "
                f"the number of attention heads ({heads})"
            )
        if local_model not in {"gin", "gine"}:
            raise ValueError("gps_local_model must be either 'gin' or 'gine'")

        self.local_model = local_model
        self.input_projection = (
            nn.Identity()
            if in_dims == hidden_dims
            else nn.Linear(in_dims, hidden_dims)
        )

        if local_model == "gine":
            # ZINC stores bond types as integer values in {1, ..., n_edge_types}.
            self.edge_embedding = nn.Embedding(n_edge_types + 1, hidden_dims)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            local_mlp = nn.Sequential(
                nn.Linear(hidden_dims, hidden_dims),
                nn.ReLU(),
                nn.Linear(hidden_dims, hidden_dims),
            )
            if local_model == "gine":
                local_conv = GINEConv(local_mlp, train_eps=True)
            else:
                local_conv = GINConv(local_mlp, train_eps=True)

            self.layers.append(
                GPSConv(
                    channels=hidden_dims,
                    conv=local_conv,
                    heads=heads,
                    dropout=dropout,
                    attn_type="multihead",
                    attn_kwargs={"dropout": attn_dropout},
                )
            )

        pooling_name = "none" if pooling is None else str(pooling).lower()
        if pooling_name == "add":
            self.pooling = global_add_pool
        elif pooling_name == "mean":
            self.pooling = global_mean_pool
        elif pooling_name == "none":
            self.pooling = None
        else:
            raise ValueError("GPS pooling must be 'add', 'mean', or 'none'")

        self.mlp = create_mlp(hidden_dims, target_dim)

    def forward(
        self,
        X_n: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        PE: Optional[torch.Tensor],
        snorm: Optional[torch.Tensor],
        batch: torch.Tensor,
    ) -> torch.Tensor:
        del PE, snorm  # DIAL has already been fused into X_n by the wrapper.

        X_n = self.input_projection(X_n)

        if self.local_model == "gine":
            if edge_attr is None:
                raise ValueError("The GINE GPS local model requires edge attributes")
            edge_attr = self.edge_embedding(edge_attr.view(-1).long())

        for layer in self.layers:
            if self.local_model == "gine":
                X_n = layer(
                    X_n,
                    edge_index,
                    batch=batch,
                    edge_attr=edge_attr,
                )
            else:
                X_n = layer(X_n, edge_index, batch=batch)

        if self.pooling is None:
            # Node-level prediction, as used by cycle counting.
            return self.mlp(X_n)

        # Graph-level prediction, as used by ZINC.
        X_n = self.pooling(X_n, batch)
        prediction = self.mlp(X_n)
        return prediction.squeeze(dim=-1) if prediction.size(-1) == 1 else prediction
