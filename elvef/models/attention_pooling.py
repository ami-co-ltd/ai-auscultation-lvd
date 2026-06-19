#!/usr/bin/env python

import torch
from torch import nn


class MultiHeadAttentionPooling(nn.Module):
    """Multi-Head Attention Pooling layer."""

    def __init__(self, n_head=1, n_hidden=768) -> None:
        super().__init__()

        self.n_head: int = n_head
        self.n_hidden: int = n_hidden
        self.linear_head = nn.Linear(n_hidden, n_head)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Compute scaled dot product attention.

        Note:
            - B: batch size
            - T: sequence length of query
            - C: dimension of input
            - H: the number of heads

        Args:
            input_data (torch.Tensor): input data of shape (B, C, T)

        Returns:
            torch.Tensor: Output tensor (B, HxC, 1)

        """
        n_batch = input_data.size(0)
        input_data = input_data.transpose(1, 2)  # -> (B, T, C)

        # compute attention weights
        # (B, T, C) -> (B, T, H)
        weights = self.linear_head(input_data).view(n_batch, -1, self.n_head)
        weights = weights.transpose(1, 2)  # -> (B, H, T)

        # apply weights
        # (B, H, T) x (B, T, C)
        output = torch.matmul(weights, input_data).view(n_batch, -1, 1)
        return output
