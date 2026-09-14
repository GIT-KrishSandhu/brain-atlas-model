from typing import Tuple, List, Union, Type
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


def maybe_convert_scalar_to_list(conv_op: Type[_ConvNd], scalar_or_list):
    if not isinstance(scalar_or_list, (tuple, list)):
        dim = 3 if issubclass(conv_op, nn.Conv3d) else 2
        return [scalar_or_list] * dim
    return list(scalar_or_list)


def get_matching_convtransp(conv_op: Type[_ConvNd] = nn.Conv3d):
    if conv_op == nn.Conv3d:
        return nn.ConvTranspose3d
    elif conv_op == nn.Conv2d:
        return nn.ConvTranspose2d
    elif conv_op == nn.Conv1d:
        return nn.ConvTranspose1d
    else:
        raise ValueError(f"Unknown conv_op: {conv_op}")


def get_matching_pool_op(conv_op: Type[_ConvNd] = nn.Conv3d, adaptive: bool = False, pool_type: str = 'avg'):
    if conv_op == nn.Conv3d:
        if pool_type == 'avg':
            return nn.AdaptiveAvgPool3d if adaptive else nn.AvgPool3d
        elif pool_type == 'max':
            return nn.AdaptiveMaxPool3d if adaptive else nn.MaxPool3d
    elif conv_op == nn.Conv2d:
        if pool_type == 'avg':
            return nn.AdaptiveAvgPool2d if adaptive else nn.AvgPool2d
        elif pool_type == 'max':
            return nn.AdaptiveMaxPool2d if adaptive else nn.MaxPool2d
    raise ValueError(f"Unsupported combination: conv_op={conv_op}, pool_type={pool_type}")


class ConvDropoutNormReLU(nn.Module):
    def __init__(
        self,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        stride: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict = None,
        nonlin_first: bool = False
    ):
        super(ConvDropoutNormReLU, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride
        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)

        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        ops = []
        self.conv = conv_op(
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding=[(i - 1) // 2 for i in kernel_size],
            dilation=1,
            bias=conv_bias
        )
        ops.append(self.conv)

        if dropout_op is not None and dropout_op_kwargs is not None:
            self.dropout = dropout_op(**dropout_op_kwargs)
            ops.append(self.dropout)

        if norm_op is not None:
            self.norm = norm_op(output_channels, **norm_op_kwargs)
            ops.append(self.norm)

        if nonlin is not None:
            self.nonlin = nonlin(**nonlin_kwargs)
            ops.append(self.nonlin)

        if nonlin_first and (norm_op is not None and nonlin is not None):
            ops[-1], ops[-2] = ops[-2], ops[-1]

        self.all_modules = nn.Sequential(*ops)

    def forward(self, x):
        return self.all_modules(x)


class BasicBlockD(nn.Module):
    def __init__(
        self,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        stride: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict = None,
        stochastic_depth_p: float = 0.0,
        squeeze_excitation: bool = False,
        squeeze_excitation_reduction_ratio: float = 1.0 / 16
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride
        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)

        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        self.conv1 = ConvDropoutNormReLU(
            conv_op, input_channels, output_channels, kernel_size, stride, conv_bias,
            norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
        )
        self.conv2 = ConvDropoutNormReLU(
            conv_op, output_channels, output_channels, kernel_size, 1, conv_bias,
            norm_op, norm_op_kwargs, None, None, None, None
        )
        self.nonlin2 = nonlin(**nonlin_kwargs) if nonlin is not None else lambda x: x

        has_stride = (isinstance(stride, int) and stride != 1) or any([i != 1 for i in stride])
        requires_projection = (input_channels != output_channels)

        if has_stride or requires_projection:
            ops = []
            if has_stride:
                ops.append(get_matching_pool_op(conv_op=conv_op, adaptive=False, pool_type='avg')(stride, stride))
            if requires_projection:
                ops.append(
                    ConvDropoutNormReLU(
                        conv_op, input_channels, output_channels, 1, 1, False,
                        norm_op, norm_op_kwargs, None, None, None, None
                    )
                )
            self.skip = nn.Sequential(*ops)
        else:
            self.skip = lambda x: x

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv2(self.conv1(x))
        out = out + residual
        return self.nonlin2(out)


class StackedResidualBlocks(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        initial_stride: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict = None,
        block: Type[nn.Module] = BasicBlockD
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            block(
                conv_op, input_channels, output_channels, kernel_size, initial_stride,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                nonlin, nonlin_kwargs
            ),
            *[
                block(
                    conv_op, output_channels, output_channels, kernel_size, 1,
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                    nonlin, nonlin_kwargs
                )
                for _ in range(1, n_blocks)
            ]
        )

    def forward(self, x):
        return self.blocks(x)


class StackedConvBlocks(nn.Module):
    def __init__(
        self,
        num_convs: int,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: Union[int, List[int], Tuple[int, ...]],
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        initial_stride: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict = None,
        nonlin_first: bool = False
    ):
        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        self.convs = nn.Sequential(
            ConvDropoutNormReLU(
                conv_op, input_channels, output_channels[0], kernel_size, initial_stride,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                nonlin, nonlin_kwargs, nonlin_first
            ),
            *[
                ConvDropoutNormReLU(
                    conv_op, output_channels[i - 1], output_channels[i], kernel_size, 1,
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                    nonlin, nonlin_kwargs, nonlin_first
                )
                for i in range(1, num_convs)
            ]
        )
        self.output_channels = output_channels[-1]

    def forward(self, x):
        return self.convs(x)
