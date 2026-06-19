#!/usr/bin/env python


import dataclasses


@dataclasses.dataclass
class MultiHeadAttentionPoolingParam:
    """dataclass for MultiHeadAttentionPooling

    Attributes:
        n_head (int): the number of heads
        n_feat (int): input feature dimension
        dropout_rate (float): dropout rate
    """

    n_head: int
    n_feat: int
    dropout_rate: float

    def __post_init__(self) -> None:
        """init"""
        self.check_parameters()

    def check_parameters(self) -> None:
        """check_parameters

        Raises:
            ValueError: invalid head number
            ValueError: invalid feature dimension
            ValueError: invalid dropout rate

        """
        if self.n_head <= 0:
            raise ValueError("Invalid head number")

        if self.n_feat <= 0:
            raise ValueError("Invalid feature dimension")

        if not 0 <= self.dropout_rate < 1:
            raise ValueError("Invalid dropout rate")
