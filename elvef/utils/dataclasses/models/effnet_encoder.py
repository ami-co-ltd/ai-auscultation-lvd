#!/usr/bin/env python
# pylint: disable=too-many-instance-attributes

import dataclasses
import math
import re
from typing import List, Tuple

import torch


@dataclasses.dataclass
class EffnetGlobalParam:
    """parameters for EfficientNet

    Attributes:
        width_coefficient (float): parameter to control CNN width
        depth_coefficient (float): parameter to control CNN depth
        depth_divisor (int): parameter to control CNN depth size
        image_size (int): shape of input data  # (=sequence length in 1D CNN)
        dropout_rate (float): dropout rate
        activation_function_name (str): activation function name
        batch_norm_momentum (float): batch norm param (not used)
        batch_norm_epsilon (float): batch norm param (not used)

    """

    width_coefficient: float
    depth_coefficient: float
    depth_divisor: int

    image_size: int
    dropout_rate: float

    activation_function_name: str

    batch_norm_momentum: float = 0.99
    batch_norm_epsilon: float = 1e-3

    def __post_init__(self) -> None:
        """init"""
        self.check_parameters()

    def check_parameters(self) -> None:
        """check parameters

        Raises:
            ValueError: width/depth coefficient should be a positive number.
            ValueError: depth divisor should be a positive value
            ValueError: not (0 <= dropout_rate <=1)
            ValueError: Got unsupported activation function name
            ValueError: Got unexpected batch_normalization params.
        """
        # 1. width coefficient
        if (self.width_coefficient <= 0) or (self.depth_coefficient <= 0):
            raise ValueError("width/depth coefficient should be a positive number.")

        # 2. depth_divisor
        if self.depth_divisor <= 0:
            raise ValueError("depth divisor should be a positive value")

        # 3. dropout_rate
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("0 <= dropout_rate <=1")

        # 4. act_fn_name
        if self.activation_function_name not in ["mish", "relu"]:
            raise ValueError("Got unsupported activation function name")

        # 5. batch normalization
        if (self.batch_norm_momentum <= 0) or (self.batch_norm_epsilon < 0):
            raise ValueError("Got unexpected batch_normalization params.")

    def round_filters(self, n_filters: int) -> int:
        """Calculate and round number of filters based on width multiplier.

        Args:
            n_filters (int): filters number to be calculated.

        Returns:
            int: New filters number after calculating.
        """
        n_filters = int(n_filters * self.width_coefficient)

        # update n_filters
        # follow the formula transferred from official TensorFlow implementation
        new_filters = max(
            self.depth_divisor,
            int(n_filters + self.depth_divisor / 2)
            // self.depth_divisor
            * self.depth_divisor,
        )

        # if new_filters < 0.9 * n_filters:  # prevent rounding by more than 10%
        # new_filters += self.depth_divisor
        return int(new_filters)

    def round_repeats(self, repeats: int) -> int:
        """Calculate module's repeat number of a block based on depth multiplier.
          Use depth_coefficient.

        Args:
            repeats (int): num_repeat to be calculated.

        Returns:
            int: New repeat number after calculating.
        """
        # multiplier = global_params.depth_coefficient
        # if not multiplier:
        # return repeats

        # follow the formula transferred from official TensorFlow implementation
        return int(math.ceil(self.depth_coefficient * repeats))

    def make_batch_norm(
        self, n_dimension: int
    ) -> torch.nn.modules.batchnorm.BatchNorm1d:
        """create BatchNorm1d instance

        Args:
            n_dimension (int): dimension

        Returns:
            torch.nn.modules.batchnorm.BatchNorm1d: instance
        """
        return torch.nn.BatchNorm1d(
            n_dimension,
            momentum=1.0 - self.batch_norm_momentum,
            eps=self.batch_norm_epsilon,
        )


@dataclasses.dataclass
class EffnetBlockParam:
    """parameters of MBConv block

    Attributes:
        num_repeat (int): repeat number
        kernel_size (int): kernel size
        stride (Tuple[int]): stride size
        expand_ratio (float): expansion ratio
        input_filters (int): input channel size
        output_filters (int): output channel size
        se_ratio (float): reduction factor of squeeze-and-excitation (SE) module
        has_skip_connect (bool): has skip-command if True
            no 'no-skip' in str_block_args.
            Defaults to True.
    """

    num_repeat: int
    kernel_size: int
    stride: Tuple[int]
    expand_ratio: float
    input_filters: int
    output_filters: int
    se_ratio: float
    has_skip_command: bool = True

    def __post_init__(self) -> None:
        """init"""
        self.check_parameters()

    def check_parameters(self) -> None:
        """check parameters

        Raises:
            ValueError: Invalid num_repeat
            ValueError: Invalid kernel or stride size
            ValueError: Got negative expand _ratio

        """
        # 1. num_repeat
        if self.num_repeat <= 0:
            raise ValueError("Invalid num_repeat")

        # 2. kernel size and stride size
        if (self.kernel_size <= 0) or (self.stride[0] <= 0):
            raise ValueError("Invalid kernel or stride size")

        # 3. expand ratio / SE ratio
        if self.expand_ratio <= 0:
            raise ValueError("Got negative expand _ratio")

        # 4. squeeze-and-excitation ratio
        # (MEMO): se_ratio accepts
        # - negative value
        # - value greater than 1

        # 5. IO filters
        if (self.input_filters <= 0) or (self.output_filters <= 0):
            raise ValueError("Got negative IO filter number")

    def update_parameters(
        self,
        input_filters: int,
        output_filters: int,
        num_repeat: int = -1,
        stride: Tuple[int] = (-1,),
    ) -> None:
        """update some parameters

        Args:
            input_filters (int): input filters
            output_filters (int): output filters
            num_repeat (int): num repeats
            stride (Tuple[int]): stride
        """
        self.input_filters = input_filters
        self.output_filters = output_filters

        if num_repeat >= 0:
            self.num_repeat = num_repeat
        if stride[0] > 0:
            self.stride = stride

        self.check_parameters()

    @property
    def has_se_module(self) -> bool:
        """check squeeze-and-excitation (SE) module is activated

        Returns:
            bool: use SE module if True
        """
        if 0 < self.se_ratio <= 1:
            return True
        return False

    @property
    def use_skip_connect(self) -> bool:
        """check that the MBConvBlock uses skip-connection

        Returns:
            bool: use skip-connection if True
        """
        if (
            self.has_skip_command
            and (self.stride[0] == 1)
            and (self.input_filters == self.output_filters)
        ):
            return True
        return False


@dataclasses.dataclass
class EfficientNetParam:
    """parameters for EfficientNet

    Attributes:
        global_param (EffnetGlobalParam): global parameters
        stem_param (Tuple[int, int, Tuple[int]]):
            output channel size, kernel, and stride size of stem cnn.
        str_block_args (List[str]): user-defined network architecture
        block_args (List[EffnetBlockParam]): decoded block args
    """

    # - global
    global_param: EffnetGlobalParam

    # - stem
    # stem_output_channels: int
    # stem_kernel_size: int
    # stem_stride_size: int
    stem_param: Tuple[int, int, Tuple[int]]

    # - blocks
    str_block_args: List[str]
    block_args: List[EffnetBlockParam] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """init"""
        self.check_parameters()
        self.decode_block_args()
        self.check_chain()

    def check_parameters(self) -> None:
        """check parameters

        Raises:
            ValueError: Invalid stem CNN parameter
            ValueError: block args is an empty list
        """

        # 1. stem related
        if (
            (self.stem_param[0] <= 0)
            or (self.stem_param[1] <= 0)
            or (self.stem_param[2][0] <= 0)
        ):
            raise ValueError("Invalid stem CNN parameter")

        # 2. block_args
        if len(self.str_block_args) == 0:
            raise ValueError("block args is an empty list")

    @staticmethod
    def decode_block_string(block_string: str) -> EffnetBlockParam:
        """Get a block through a string notation of arguments.

        Args:
            block_string (str): A string notation of arguments.
                                Examples: 'r1_k3_s11_e1_i32_o16_se0.25_noskip'.

        Raises:
            KeyError: options has no keys (r, k, s, e, i, o, se)

        Returns:
            EffnetBlockParam: populated EffnetBlockParam instance
        """
        elms = block_string.split("_")

        options = {}
        for elm in elms:
            splits = re.split(r"(\d.*)", elm)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        return EffnetBlockParam(
            num_repeat=int(options["r"]),
            kernel_size=int(options["k"]),
            stride=(int(options["s"]),),
            expand_ratio=int(options["e"]),
            input_filters=int(options["i"]),
            output_filters=int(options["o"]),
            se_ratio=float(options["se"]) if "se" in options else -1,
            has_skip_command=("noskip" not in block_string),
        )

    def decode_block_args(self) -> None:
        """Decode a list of string notations to specify blocks inside the network"""

        self.block_args = []

        for block_string in self.str_block_args:
            self.block_args.append(EfficientNetParam.decode_block_string(block_string))

    def check_chain(self) -> None:
        """check filter size

        Raises:
            ValueError: if there's channel size inconsistency.
        """

        if self.stem_param[0] != self.block_args[0].input_filters:
            raise ValueError("Got filter size inconsistency")

        for i in range(1, len(self.block_args)):
            if (
                self.block_args[i - 1].output_filters
                != self.block_args[i].input_filters
            ):
                raise ValueError("Got filter size inconsisitency")
