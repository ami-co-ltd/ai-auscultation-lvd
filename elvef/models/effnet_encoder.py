#!/usr/bin/env python
# pylint: disable=no-member,arguments-renamed,too-many-instance-attributes

import math
from functools import partial
from logging import getLogger
from typing import Callable, Tuple

import omegaconf
import torch
import torch.nn.functional as F
from torch import nn

from elvef.models.base_extractor import BaseExtractor
from elvef.models.define_activation import create_activation
from elvef.utils.dataclasses.models.effnet_encoder import (
    EfficientNetParam,
    EffnetGlobalParam,
)

logger = getLogger(__name__)


def calculate_output_image_size(input_image_size: int, stride: Tuple[int]) -> int:
    """Calculates the output image size when using Conv2dSamePadding with a stride.
       Necessary for static padding. Thanks to mannatsingh for pointing this out.

    Args:
        input_image_size (int): Size of input image.
        stride (Tuple[int]): Conv1d operation's stride.

    Returns:
        int: output image_size after convolution (w/ given stride size)
    """
    image_width = int(math.ceil(input_image_size / stride[0]))
    return image_width


class Conv1dStaticSamePadding(nn.Conv1d):
    """1D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
      The padding mudule is calculated in construction function, then used in forward.

    Attributes:
        stride (Tuple[int]): stride size
        static_padding_val (int): padding size
        static_padding (Union[torch.nn.modules.linear.Identity, None]):
            padding function
    """

    def __init__(
        self,
        io_channels: Tuple[int, int],
        kernel_size: int,
        stride: Tuple[int],
        image_size: int,
        **kwargs,
    ):
        """init

        Args:
            io_channels (Tuple[int, int]): input and output channel size
            kernel_size (int): kernel size
            stride (Tuple[int]): stride size
            image_size (int): image size
            kwargs: extra keyword arguments passed to nn.Conv1d
        """
        super().__init__(io_channels[0], io_channels[1], kernel_size, stride, **kwargs)

        self.stride = stride

        image_width = image_size
        kernel_width = self.weight.size()[-1]  # size: (out_channs, in/group, kW)
        output_width = math.ceil(image_width / self.stride[0])

        # compute pad size in
        # https://pytorch.org/docs/1.9.1/generated/torch.nn.Conv1d.html
        pad_width = max(
            (output_width - 1) * self.stride[0]
            + (kernel_width - 1) * self.dilation[0]
            + 1
            - image_width,
            0,
        )

        if pad_width > 0:
            # pad image
            self.static_padding_val = (pad_width // 2, pad_width - pad_width // 2)
            self.static_padding = None
        else:
            self.static_padding = nn.Identity()

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """forward path

        Note:
             - B: batch size
             - C, C': channel dimension
             - T, T': sequence length

        Args:
            input_data (torch.Tensor): input tensor of shape (B, C, T).

        Returns:
            torch.Tensor: output of this model after processing,
                          shape: (B, C', T').

        """
        if self.static_padding is None:
            data = F.pad(
                input_data, self.static_padding_val
            )  # mode: constant, value: 0
        else:
            data = self.static_padding(input_data)

        data = F.conv1d(
            data,
            self.weight,
            self.bias,
            self.stride[0],
            self.padding,
            self.dilation,
            self.groups,
        )
        return data


def create_same_padding_conv1d(image_size: int) -> Callable:
    """create Conv1dStaticSamePadding instance using given image_size

    Args:
        image_size (int): sequence length

    Returns:
        Callable: Conv1dStaticSamePadding
    """
    return partial(Conv1dStaticSamePadding, image_size=image_size)


#
# MBConv Module
#
class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.

    Attributes:
        _conv1 (Conv1dStaticSamePadding): first convolution
            (fused depth-wise and separable convolution)
        _batch_norm_1 (torch.nn.modules.batchnorm.BatchNorm1d):
            batch normalization after first convolution
        has_se_module (bool): apply squeeze-and-excitation if True
        _se_reduce (Conv1dStaticSamePadding): convolution for squeeze
        _se_expand (Conv1dStaticSamePadding): convolution for excitation
        _project_conv (Conv1dStaticSamePadding): point-wise convolution module
        _batch_norm2 (torch.nn.modules.batchnorm.BatchNorm1d):
            batch normalization module after squeeze-and-excitation
        _activation (ActivationType): activation function
        _dropout (torch.nn.modules.dropout.Dropout): dropout module
        use_skip_connect (bool): use skip_connection if True

    """

    def __init__(
        self,
        block_param,
        global_param: EffnetGlobalParam,
        image_size: int,
    ) -> None:
        """init

        Args:
            block_param (EffnetBlockParam): arguments for this MBConvBlock
            global_param (EffnetGlobalParam): global parameters
            image_size (int): image size
        """
        # 1.
        super().__init__()

        # 2. first Convolution
        input_channels = block_param.input_filters
        output_channels = int(block_param.input_filters * block_param.expand_ratio)
        conv1d = create_same_padding_conv1d(image_size=image_size)
        self._conv1 = conv1d(
            (input_channels, output_channels),
            kernel_size=block_param.kernel_size,
            stride=block_param.stride,
            bias=False,
        )
        self._batch_norm1 = global_param.make_batch_norm(output_channels)
        image_size = calculate_output_image_size(image_size, block_param.stride)

        # 3. Squeeze and Excitation layer, if desired
        self.has_se_module = block_param.has_se_module
        if self.has_se_module:
            conv1d = create_same_padding_conv1d(image_size=1)
            num_squeezed_channels = max(
                1, int(block_param.input_filters * block_param.se_ratio)
            )
            self._se_reduce = conv1d(
                (output_channels, num_squeezed_channels),
                kernel_size=1,
                stride=(1,),
            )
            self._se_expand = conv1d(
                (num_squeezed_channels, output_channels),
                kernel_size=1,
                stride=(1,),
            )

        # 4. Pointwise convolution
        final_output_channels = block_param.output_filters
        conv1d = create_same_padding_conv1d(image_size=image_size)
        self._project_conv = conv1d(
            (output_channels, final_output_channels),
            kernel_size=1,
            stride=(1,),
            bias=False,
        )
        self._batch_norm2 = global_param.make_batch_norm(final_output_channels)

        # 5. dropout, etc
        self._activation = create_activation(global_param.activation_function_name)
        self._dropout = nn.Dropout(global_param.dropout_rate)
        # whether to use skip connection
        self.use_skip_connect = block_param.use_skip_connect

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """MBConvBlock's forward function.

        Args:
            input_data (torch.Tensor): input tensor.

        Returns:
            Output of this block after processing.
        """
        data = input_data

        # 1. first conv
        data = self._conv1(data)
        data = self._batch_norm1(data)
        data = self._activation(data)

        # 2. Squeeze and Excitation
        if self.has_se_module:
            data_squeezed = F.adaptive_avg_pool1d(data, 1)
            data_squeezed = self._se_reduce(data_squeezed)
            data_squeezed = self._activation(data_squeezed)
            data_squeezed = self._se_expand(data_squeezed)
            data = torch.sigmoid(data_squeezed) * data

        # 3. Pointwise Convolution
        data = self._project_conv(data)
        data = self._batch_norm2(data)

        # 4. Skip connection and drop connect
        if self.use_skip_connect:
            data = self._dropout(data)
            data = data + input_data  # skip connection

        return data


class EfficientNet(nn.Module):
    """EfficientNet model

    References:
        - [paper](https://arxiv.org/abs/1905.11946) (EfficientNet)

    Attributes:
        param (EfficientNetParam): parameters for EfficientNet
        extractor (BaseExtractor): feature extraction module
        _conv_stem (Conv1dStaticSamePadding): stem CNN layer
        _batch_norm0 (torch.nn.modules.batchnorm.BatchNorm1d):
            batch normalization module used after stem cnn.
        _blocks (torch.nn.modules.container.ModuleList): MBConv blocks
        output_dim (int): output dimension
        _conv_head (Conv1dStaticSamePadding): last convolution layer
        _batch_norm1 (torch.nn.modules.batchnorm.BatchNorm1d):
            batch normalization after conv_head layer
        _activation (ActivationType): activation

    """

    def __init__(
        self,
        input_channels: int,
        param: EfficientNetParam,
        output_dim: int,
        extractor: BaseExtractor,
    ):
        """init

        Args:
            input_channels (int): input dimension of EfficientNet
            param (EfficientNet): parameters for EfficientNet
            output_dim (int): output dimension of EfficientNet (if > 0)
            extractor (BaseExtractor): feature extraction module
        """

        # 1.
        super().__init__()
        self.param = param
        self.extractor = extractor

        # 2. stem
        self._conv_stem, self._batch_norm0, image_size = self.make_stem_cnn(
            self.param.global_param.image_size,
            input_channels,
        )

        # 3. Build MBConvBlocks
        self._blocks = nn.ModuleList([])
        image_size, input_channels = self.make_mbconv_blocks(image_size, param)

        # 4. last layer
        # use default setting for now
        self.output_dim = (
            output_dim if output_dim > 0 else param.global_param.round_filters(512)
        )
        conv1d = create_same_padding_conv1d(image_size=image_size)
        self._conv_head = conv1d(
            (input_channels, self.output_dim),
            kernel_size=1,
            bias=False,
            stride=(1,),
        )
        self._batch_norm1 = param.global_param.make_batch_norm(self.output_dim)

        # 5. other settings
        self._activation = create_activation(
            param.global_param.activation_function_name
        )

    def make_stem_cnn(
        self,
        image_size: int,
        input_channels: int,
    ) -> Tuple[Conv1dStaticSamePadding, torch.nn.modules.batchnorm.BatchNorm1d, int]:
        """make first CNN layer

        Args:
            image_size (int): sequence length of input tensor
            input_channels (int): input channel size

        Returns:
            Tuple[Conv1dStaticSamePadding, torch.nn.modules.batchnorm.BatchNorm1d, int]:
                stem, batch norm, and updated image_size
        """
        global_param = self.param.global_param

        conv1d = create_same_padding_conv1d(image_size=image_size)
        output_channels = global_param.round_filters(self.param.stem_param[0])

        return (
            conv1d(
                (input_channels, output_channels),
                kernel_size=self.param.stem_param[1],
                stride=self.param.stem_param[2],
                bias=False,
            ),
            global_param.make_batch_norm(output_channels),
            calculate_output_image_size(image_size, self.param.stem_param[2]),
        )

    def make_mbconv_blocks(
        self, image_size: int, param: EfficientNetParam
    ) -> Tuple[int, int]:
        """make list of MBConv modules

        Args:
            image_size (int): sequence length of input tensor
            param (EfficientNetParam): parameter

        Returns:
            Tuple[int, int]: updated image_size and channel size of next Conv1d
        """
        global_param = param.global_param
        block_param = param.block_args[0]  # undefined-loop-variable

        for block_param in param.block_args:
            # 1. update block input and output filters based on depth multiplier.
            block_param.update_parameters(
                global_param.round_filters(block_param.input_filters),
                global_param.round_filters(block_param.output_filters),
                global_param.round_repeats(block_param.num_repeat),
            )

            # 2. the first block needs to take care of stride and filter size increase.
            self._blocks.append(
                MBConvBlock(
                    block_param,
                    global_param,
                    image_size=image_size,
                )
            )
            image_size = calculate_output_image_size(image_size, block_param.stride)

            # 3. modify block_args to keep same output size
            if block_param.num_repeat > 1:
                block_param.update_parameters(
                    block_param.output_filters,
                    block_param.output_filters,
                    stride=(1,),
                )

            # 4. stack extra layers
            for _ in range(block_param.num_repeat - 1):
                self._blocks.append(
                    MBConvBlock(
                        block_param,
                        global_param,
                        image_size=image_size,
                    )
                )

        return image_size, block_param.output_filters

    @staticmethod
    def help():
        """dump help, usage, warning, etc."""
        logger.warning("")

    @staticmethod
    def create_effnet_arguments(
        input_seqlen: int,
        config: omegaconf.dictconfig.DictConfig,
    ) -> EfficientNetParam:
        """get arguments for EfficientNet

        Args:
            input_seqlen (int): sequence length of input data
            config (omegaconf.dictconfig.DictConfig): DictConfig

        Raises:
            ValueError: failed to parse EfficientNetParam

        Returns:
            EfficientNetParam: created instance

        """
        try:
            width_coefficient = float(config["model"]["encoder"]["width_coefficient"])
            depth_coefficient = float(config["model"]["encoder"]["depth_coefficient"])
            depth_divisor = int(config["model"]["encoder"]["depth_divisor"])

            dropout_rate = float(config["model"]["encoder"]["dropout_rate"])

            stem_output_channels = int(
                config["model"]["encoder"]["stem_output_channels"]
            )
            stem_kernel_size = int(config["model"]["encoder"]["stem_kernel_size"])
            stem_stride_size = int(config["model"]["encoder"]["stem_stride_size"])

            block_args = list(config["model"]["encoder"]["block_args"])

            activation_function_name = str(
                config["model"]["encoder"]["activation_function_name"]
            )

        except ValueError as exit_after_usage:
            EfficientNet.help()
            raise ValueError("failed to parse EfficientNet") from exit_after_usage

        global_param = EffnetGlobalParam(
            width_coefficient,
            depth_coefficient,
            depth_divisor,
            input_seqlen,
            dropout_rate,
            activation_function_name,
        )

        return EfficientNetParam(
            global_param,
            (stem_output_channels, stem_kernel_size, (stem_stride_size,)),
            block_args,
        )

    def get_output_dim(self) -> int:
        """get hidden vector dimension of EfficientNet encoder

        Returns:
            int: dimension
        """
        return self.output_dim

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """EfficientNet's forward function.

        Note:
             - B: batch size
             - C, C': channel dimension
             - T, T': sequence length

        Args:
            input_data (torch.Tensor): input tensor of shape (B, C, T).

        Returns:
            torch.Tensor: output of this model after processing,
                          shape: (B, C', T').

        """
        # feature extraction
        input_data = self.extractor(input_data)

        # stem
        data = self._activation(self._batch_norm0(self._conv_stem(input_data)))

        # blocks
        for block in self._blocks:
            data = block(data)

        data = self._activation(self._batch_norm1(self._conv_head(data)))
        return data


def create_effnet1d(
    input_channels: int,
    input_seqlen: int,
    config: omegaconf.dictconfig.DictConfig,
    extractor: BaseExtractor,
    output_dim: int = -1,
) -> EfficientNet:
    """create EfficientNet instance

    Args:
        input_channels (int): input channels
        input_seqlen (int): sequence length of input tensor
        config (omegaconf.dictconfig.DictConfig): configuration
        extractor (BaseExtractor): feature extraction module
        output_dim (int): output dimension (if needed). Defaults to -1.

    Returns:
        EfficientNet: network
    """
    param = EfficientNet.create_effnet_arguments(input_seqlen, config)
    return EfficientNet(input_channels, param, output_dim, extractor)
