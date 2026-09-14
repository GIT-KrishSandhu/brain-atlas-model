from typing import Tuple, Union, List, Type
import torch
import torch.nn as nn
from src.models.p3.blocks import (
    StackedResidualBlocks,
    BasicBlockD,
    StackedConvBlocks,
    get_matching_convtransp
)
from src.models.p3.head import ClassificationHead


class P3Architecture(nn.Module):
    """
    Independent recreation of the official P3 multi-task architecture:
    ResEncoderUNet_two_seg_with_cls_modality.

    Contains:
      - 6-stage 3D residual encoder with bottleneck at stage 5 ([B, 320, 7, 7, 7]).
      - Multi-head cross-attention pooling for presence, location, and modality classification.
      - Dual 3D U-Net decoders with skip connections for vessel and territory segmentation.
      - Deep supervision outputs.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels_1: int = 15,
        out_channels_2: int = 14,
        cls_head_num_classes_list: List[int] = [1, 13],
        cls_drop_out_list: List[float] = [0.0, 0.0],
        cls_query_num_list: List[int] = [2, 16],
        use_cross_attention: bool = True,
        n_stages: int = 6,
        features_per_stage: List[int] = [32, 64, 128, 256, 320, 320],
        kernel_sizes: List[Union[Tuple[int, int, int], List[int]]] = [[3, 3, 3]] * 6,
        strides: List[Union[Tuple[int, int, int], List[int]]] = [[1, 1, 1]] + [[2, 2, 2]] * 5,
        dropout_rate: float = 0.0,
        deep_supervision: bool = True,
        norm_op: Type[nn.Module] = nn.InstanceNorm3d,
        norm_op_kwargs: dict = {"eps": 1e-05, "affine": True},
        conv_op: Type[nn.Module] = nn.Conv3d,
        conv_bias: bool = True,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = {"inplace": True},
        dropout_op: Union[None, Type[nn.Module]] = None,
        dropout_op_kwargs: dict = None,
        n_blocks_per_stage: List[int] = [1, 3, 4, 6, 6, 6],
        n_conv_per_stage_decoder: List[int] = [1, 1, 1, 1, 1],
        nonlin_first: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels_1 = out_channels_1
        self.out_channels_2 = out_channels_2
        self.deep_supervision = deep_supervision
        self.num_stages = len(features_per_stage)

        # 1. Convolutional Encoder Blocks (U-Net style)
        self.conv_encoder_blocks = self.build_encoder_block(
            n_blocks_per_stage, features_per_stage, kernel_sizes, strides, in_channels,
            conv_op, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
            nonlin, nonlin_kwargs, BasicBlockD
        )

        transpconv_op = get_matching_convtransp(conv_op=conv_op)

        # 2. Decoder Blocks (Dual Decoders)
        (
            self.transpconvs,
            self.transpconvs_last_two_1,
            self.transpconvs_last_two_2,
            self.decoder_blocks,
            self.decoder_blocks_last_two_1,
            self.decoder_blocks_last_two_2,
            self.seg_layers_1,
            self.seg_layers_2
        ) = self.build_decoder_blocks(
            transpconv_op, features_per_stage, kernel_sizes, strides, n_conv_per_stage_decoder,
            conv_op, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
            nonlin, nonlin_kwargs, nonlin_first
        )

        # 3. Multi-task Classification Heads
        self.cls_head_list = nn.ModuleList()
        self.cls_drop_out_list = cls_drop_out_list
        self.cls_query_num_list = cls_query_num_list

        for i in range(len(cls_head_num_classes_list)):
            cls_head_num_classes = cls_head_num_classes_list[i]
            cls_drop_out = self.cls_drop_out_list[i]
            cls_query_num = self.cls_query_num_list[i]

            self.cls_head_list.append(ClassificationHead(
                embed_dim=features_per_stage[-1],
                query_num=cls_query_num,
                num_classes=cls_head_num_classes,
                dropout=cls_drop_out,
                use_cross_attention=use_cross_attention,
                num_heads=4
            ))

        self.cls_modality_head = ClassificationHead(
            embed_dim=features_per_stage[-1],
            query_num=4,
            num_classes=4,
            dropout=0.0,
            use_cross_attention=use_cross_attention,
            num_heads=4
        )

    def build_encoder_block(
        self, n_blocks_per_stage, features_per_stage, kernel_sizes, strides, in_channels,
        conv_op, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, block_type
    ):
        blocks = nn.ModuleList()
        blocks.append(
            StackedResidualBlocks(
                n_blocks_per_stage[0], conv_op, in_channels, features_per_stage[0], kernel_sizes[0], strides[0],
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                block=block_type
            )
        )
        for i in range(1, len(n_blocks_per_stage)):
            blocks.append(
                StackedResidualBlocks(
                    n_blocks_per_stage[i], conv_op, features_per_stage[i - 1], features_per_stage[i], kernel_sizes[i],
                    strides[i], conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                    block=block_type
                )
            )
        return blocks

    def build_decoder_blocks(
        self, transpconv_op, features_per_stage, kernel_sizes, strides, n_conv_per_stage_decoder,
        conv_op, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
    ):
        transpconvs = nn.ModuleList()
        transpconvs_last_two_1 = nn.ModuleList()
        transpconvs_last_two_2 = nn.ModuleList()
        decoder_blocks = nn.ModuleList()
        decoder_blocks_last_two_1 = nn.ModuleList()
        decoder_blocks_last_two_2 = nn.ModuleList()
        seg_layers_1 = nn.ModuleList()
        seg_layers_2 = nn.ModuleList()

        for i in range(len(features_per_stage) - 1, 0, -1):
            if i >= 5:
                transpconvs.append(
                    transpconv_op(
                        features_per_stage[i], features_per_stage[i - 1], strides[i], strides[i],
                        bias=conv_bias
                    )
                )
                decoder_blocks.append(
                    StackedConvBlocks(
                        n_conv_per_stage_decoder[i - 1], conv_op, 2 * features_per_stage[i - 1], features_per_stage[i - 1],
                        kernel_sizes[i], 1, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                        nonlin, nonlin_kwargs, nonlin_first
                    )
                )
            else:
                transpconvs_last_two_1.append(
                    transpconv_op(
                        features_per_stage[i], features_per_stage[i - 1], strides[i], strides[i],
                        bias=conv_bias
                    )
                )
                transpconvs_last_two_2.append(
                    transpconv_op(
                        features_per_stage[i], features_per_stage[i - 1], strides[i], strides[i],
                        bias=conv_bias
                    )
                )
                decoder_blocks_last_two_1.append(
                    StackedConvBlocks(
                        n_conv_per_stage_decoder[i - 1], conv_op, 2 * features_per_stage[i - 1], features_per_stage[i - 1],
                        kernel_sizes[i], 1, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                        nonlin, nonlin_kwargs, nonlin_first
                    )
                )
                decoder_blocks_last_two_2.append(
                    StackedConvBlocks(
                        n_conv_per_stage_decoder[i - 1], conv_op, 2 * features_per_stage[i - 1], features_per_stage[i - 1],
                        kernel_sizes[i], 1, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                        nonlin, nonlin_kwargs, nonlin_first
                    )
                )
            seg_layers_1.append(conv_op(features_per_stage[i - 1], self.out_channels_1, kernel_size=1, stride=1, padding=0, bias=True))
            seg_layers_2.append(conv_op(features_per_stage[i - 1], self.out_channels_2, kernel_size=1, stride=1, padding=0, bias=True))

        return (
            transpconvs,
            transpconvs_last_two_1,
            transpconvs_last_two_2,
            decoder_blocks,
            decoder_blocks_last_two_1,
            decoder_blocks_last_two_2,
            seg_layers_1,
            seg_layers_2
        )

    def forward(self, input_image, only_forward_cls: bool = False):
        conv_enc_outputs = [self.conv_encoder_blocks[0](input_image)]
        for i in range(1, len(self.conv_encoder_blocks)):
            conv_enc_outputs.append(self.conv_encoder_blocks[i](conv_enc_outputs[-1]))

        lres_input = conv_enc_outputs[-1]

        # Classification heads
        cls_pred_list = []
        for cls_head in self.cls_head_list:
            cls_pred_list.append(cls_head(lres_input))

        if only_forward_cls:
            return cls_pred_list

        cls_modality_pred = self.cls_modality_head(lres_input)

        seg_outputs_1 = []
        seg_outputs_2 = []
        total_decoder_steps = len(self.decoder_blocks + self.decoder_blocks_last_two_1)

        for s in range(total_decoder_steps):
            if s <= (total_decoder_steps - 5):
                x = self.transpconvs[s](lres_input)
                x = torch.cat((x, conv_enc_outputs[-(s + 2)]), 1)
                x = self.decoder_blocks[s](x)

                if self.deep_supervision:
                    seg_outputs_1.append(self.seg_layers_1[s](x))
                    seg_outputs_2.append(self.seg_layers_2[s](x))
                lres_input = x
            else:
                if s == (total_decoder_steps - 4):
                    lres_input_1 = lres_input
                    lres_input_2 = lres_input

                idx = s - len(self.decoder_blocks)
                x_1 = self.transpconvs_last_two_1[idx](lres_input_1)
                x_2 = self.transpconvs_last_two_2[idx](lres_input_2)
                x_1 = torch.cat((x_1, conv_enc_outputs[-(s + 2)]), 1)
                x_2 = torch.cat((x_2, conv_enc_outputs[-(s + 2)]), 1)
                x_1 = self.decoder_blocks_last_two_1[idx](x_1)
                x_2 = self.decoder_blocks_last_two_2[idx](x_2)

                if self.deep_supervision:
                    seg_outputs_1.append(self.seg_layers_1[s](x_1))
                    seg_outputs_2.append(self.seg_layers_2[s](x_2))
                elif s == (total_decoder_steps - 1):
                    seg_outputs_1.append(self.seg_layers_1[-1](x_1))
                    seg_outputs_2.append(self.seg_layers_2[-1](x_2))

                lres_input_1 = x_1
                lres_input_2 = x_2

        if self.deep_supervision:
            r_1 = seg_outputs_1[::-1]
            r_2 = seg_outputs_2[::-1]
        else:
            r_1 = seg_outputs_1[-1]
            r_2 = seg_outputs_2[-1]

        return r_1, r_2, cls_pred_list, cls_modality_pred


# Backward compatibility alias
ResEncoderUNet_two_seg_with_cls_modality = P3Architecture
