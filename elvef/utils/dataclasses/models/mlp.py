#!/usr/bin/env python

import dataclasses


@dataclasses.dataclass
class MlpParam:
    """dataclass for MLP

    Attributes:
        input_dim (int): input dimension
        hidden_dim (int): hidden dimension
        output_dim (int): output dimension
        n_layers (int): number of hidden layers
        activation_function_name (str): activation function name
    """

    input_dim: int
    hidden_dim: int
    output_dim: int
    n_layers: int
    activation_function_name: str

    def __post_init__(self) -> None:
        self.check_parameters()

    def check_parameters(self) -> None:
        """check parameters"

        Raises:
            ValueError: negative dimension for MLP
            ValueError: MLP class needs more than 1 layer(s)
            ValueError: Unsupported activation function
        """

        # dimension
        if (self.input_dim <= 0) or (self.hidden_dim <= 0) or (self.output_dim <= 0):
            raise ValueError("negative dimension for MLP")

        # layers
        if self.n_layers < 1:
            raise ValueError("MLP class needs more than 1 layer(s)")

        # activation
        if self.activation_function_name not in ["mish", "relu"]:
            raise ValueError("Unsupported activation function")
