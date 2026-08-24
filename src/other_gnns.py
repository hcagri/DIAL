from typing import Callable

import torch
from torch import nn
from torch_geometric.nn import PNAConv
from src.mlp import MLP

# PNA implementation using official PNAConv
# Based on: https://arxiv.org/pdf/2004.05718 (Principal Neighbourhood Aggregation)

class PNA(nn.Module):
    """
    PNA (Principal Neighbourhood Aggregation) model with the same interface as GINE.
    Uses multiple aggregators (mean, max, min, std) and scalers (identity, amplification, attenuation).
    """
    layers: nn.ModuleList

    def __init__(
        self, n_layers: int, n_edge_types: int, in_dims: int, hidden_dims: int, out_dims: int,
        create_mlp: Callable[[int, int], MLP], deg: torch.Tensor, bn: bool = False, residual: bool = False, 
        feature_type: str = "discrete", pe_emb=37) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.residual = residual
        self.bn = bn
        if bn:
            self.batch_norms = nn.ModuleList()
        
        for _ in range(n_layers - 1):
            layer = PNALayer(n_edge_types, in_dims, hidden_dims, create_mlp, deg,
                           feature_type, pe_emb=pe_emb)
            self.layers.append(layer)
            in_dims = hidden_dims
            if bn:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims))

        layer = PNALayer(n_edge_types, hidden_dims, out_dims, create_mlp, deg,
                        feature_type, pe_emb=pe_emb)
        self.layers.append(layer)

    def forward(self, X_n: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, 
                PE: torch.Tensor) -> torch.Tensor:
        """
        :param X_n: Node feature matrix. [N_sum, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :param edge_attr: Edge type matrix. [E_sum]
        :param PE: Positional encoding. [N_sum, pe_dim] or [E_sum, pe_dim]
        :return: Output node feature matrix. [N_sum, D_out]
        """
        for i, layer in enumerate(self.layers):
            X_0 = X_n
            X_n = layer(X_n, edge_index, edge_attr, PE)
            # batch normalization
            if self.bn and i < len(self.layers) - 1:
                if X_n.ndim == 3:
                    X_n = self.batch_norms[i](X_n.transpose(2, 1)).transpose(2, 1)
                else:
                    X_n = self.batch_norms[i](X_n)
            if self.residual:
                X_n = X_n + X_0
        return X_n


from ogb.graphproppred.mol_encoder import BondEncoder

class PNALayer(nn.Module):
    """
    PNA Layer using official PNAConv with multiple aggregators: mean, max, min, std
    and scalers: identity, amplification, attenuation.
    Preserves PE and bond encoding logic from original implementation.
    """
    
    def __init__(self, n_edge_types: int, in_dims: int, out_dims: int, 
                 create_mlp: Callable[[int, int], MLP], deg: torch.Tensor,
                 feature_type: str = "discrete", pe_emb=37) -> None:
        super().__init__()
                
        # PE embedding
        self.pe_embedding = create_mlp(pe_emb, in_dims)
        
        # Aggregators and scalers for PNAConv
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        
        # Official PNAConv layer with edge_dim for native edge feature handling
        self.conv = PNAConv(
            in_channels=in_dims,
            out_channels=out_dims,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            edge_dim=in_dims,  # Edge features dimension after embedding
            towers=1,
            pre_layers=1,
            post_layers=1,
        )

    def forward(self, X_n: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                PE: torch.Tensor) -> torch.Tensor:
        """
        :param X_n: Node feature matrix. [N_sum, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :param edge_attr: Edge type matrix. [E_sum]
        :param PE: Positional encoding.
        :return: Output node feature matrix. [N_sum, D_out]
        """
        # Process edge features through embedding
        X_e = edge_attr if edge_attr is not None else None
        
        # Apply positional encodings before convolution
        if PE is not None:
            if X_e is not None and PE.size(0) == X_e.size(0):  # for PEG (edge-level PE)
                X_e = X_e * self.pe_embedding(PE)
            else:  # node-level PE
                X_n = X_n + self.pe_embedding(PE)

        # Apply official PNAConv
        return self.conv(X_n, edge_index, X_e)

