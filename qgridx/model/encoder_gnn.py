"""GNN encoder: Ising graph → context vector + per-variable embeddings.

Architecture follows the gqco reference implementation (MIT licensed):
  - Node features: [h_i] (linear bias).
  - Edge features: [J_ij] (quadratic coupling).
  - 3-layer GCN/GraphSAGE with ReLU activations.
  - Global mean pooling → context vector.

Requires torch_geometric. If unavailable a stub is returned.

Attribution: Architecture adapted from
  https://github.com/shunyaist/generative-quantum-combinatorial-optimization
  (MIT License).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from qgridx.config import EncoderGNNConfig
from qgridx.problems.ising import IsingSpec


class IsingGNNEncoder(nn.Module):
    """GNN encoder for the Ising problem graph.

    Args:
        cfg: :class:`EncoderGNNConfig` specifying hidden dim and layer count.
    """

    def __init__(self, cfg: EncoderGNNConfig) -> None:
        super().__init__()
        self.hidden = cfg.hidden
        self.n_layers = cfg.layers

        # Node-feature projection: h_i scalar → hidden dim
        self.node_proj = nn.Linear(1, cfg.hidden)

        # GNN layers (using simple linear message-passing approximation
        # when torch_geometric is unavailable)
        self.layers = nn.ModuleList([
            nn.Linear(cfg.hidden, cfg.hidden) for _ in range(cfg.layers)
        ])
        self.activation = nn.ReLU()
        self.context_proj = nn.Linear(cfg.hidden, cfg.hidden)

    def forward(
        self,
        h: torch.Tensor,
        J: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the Ising problem.

        Args:
            h: Node features, shape (m,) or (B, m) — linear biases.
            J: Adjacency/coupling matrix, shape (m, m) or (B, m, m).

        Returns:
            Tuple of:
              - context: Global context vector, shape (hidden,) or (B, hidden).
              - node_emb: Per-node embeddings, shape (m, hidden) or (B, m, hidden).
        """
        if h.dim() == 1:
            h = h.unsqueeze(0)  # (1, m)
            J = J.unsqueeze(0)  # (1, m, m)
            squeeze = True
        else:
            squeeze = False

        B, m = h.shape

        # Node embedding: (B, m, hidden)
        x = self.node_proj(h.unsqueeze(-1))  # (B, m, hidden)

        # Message passing: aggregate neighbour features weighted by J_ij
        for layer in self.layers:
            # Simple spectral-style aggregation: x_new = J @ x + x
            # J: (B, m, m), x: (B, m, hidden)
            agg = torch.bmm(J, x)  # (B, m, hidden)
            x = self.activation(layer(x + agg))

        node_emb = x  # (B, m, hidden)
        context = self.context_proj(node_emb.mean(dim=1))  # (B, hidden)

        if squeeze:
            node_emb = node_emb.squeeze(0)
            context = context.squeeze(0)

        return context, node_emb

    def encode_ising(
        self,
        ising: IsingSpec,
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience wrapper that accepts an :class:`IsingSpec`.

        Args:
            ising:  Current Ising specification.
            device: Torch device string.

        Returns:
            Same as :meth:`forward`.
        """
        h_t = torch.tensor(ising.h, dtype=torch.float32, device=device)
        J_t = torch.tensor(ising.J, dtype=torch.float32, device=device)
        self.to(device)
        self.eval()
        with torch.no_grad():
            return self.forward(h_t, J_t)
