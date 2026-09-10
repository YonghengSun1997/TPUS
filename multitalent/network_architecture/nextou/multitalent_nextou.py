# TPUS modification: adapts NexToU to MultiTalent dataset IDs, multistems, and feature-head interfaces.
from __future__ import annotations

from typing import List, Tuple, Type, Union

import numpy as np
import torch
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp, maybe_convert_scalar_to_list
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD, StackedResidualBlocks
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from multitalent.network_architecture.nextou.encoder_decoder_blocks import OptInit, PoolGNNBlocks, SwinGNNBlocks


def _as_list(value, length: int):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * length


def _compute_img_shapes(patch_size: List[int], strides: List[List[int]], conv_op: Type[_ConvNd]):
    if conv_op == nn.Conv2d:
        h, w = int(patch_size[0]), int(patch_size[1])
        img_shape_list = [(h, w)]
        n_size_list = [h * w]
        for h_k, w_k in strides[1:]:
            h //= int(h_k)
            w //= int(w_k)
            img_shape_list.append((h, w))
            n_size_list.append(h * w)
    elif conv_op == nn.Conv3d:
        s, h, w = int(patch_size[0]), int(patch_size[1]), int(patch_size[2])
        img_shape_list = [(s, h, w)]
        n_size_list = [s * h * w]
        for s_k, h_k, w_k in strides[1:]:
            s //= int(s_k)
            h //= int(h_k)
            w //= int(w_k)
            img_shape_list.append((s, h, w))
            n_size_list.append(s * h * w)
    else:
        raise ValueError(f"unknown convolution dimensionality, conv op: {conv_op}")
    return img_shape_list, n_size_list


def _make_residual_blocks(n_blocks: int,
                          conv_op: Type[_ConvNd],
                          input_channels: int,
                          output_channels: int,
                          kernel_size,
                          initial_stride,
                          conv_bias: bool,
                          norm_op,
                          norm_op_kwargs,
                          dropout_op,
                          dropout_op_kwargs,
                          nonlin,
                          nonlin_kwargs,
                          block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                          bottleneck_channels=None,
                          stochastic_depth_p: float = 0.0,
                          squeeze_excitation: bool = False,
                          squeeze_excitation_reduction_ratio: float = 1. / 16):
    return StackedResidualBlocks(
        max(1, int(n_blocks)),
        conv_op,
        input_channels,
        output_channels,
        kernel_size,
        initial_stride,
        conv_bias,
        norm_op,
        norm_op_kwargs,
        dropout_op,
        dropout_op_kwargs,
        nonlin,
        nonlin_kwargs,
        block=block,
        bottleneck_channels=bottleneck_channels,
        stochastic_depth_p=stochastic_depth_p,
        squeeze_excitation=squeeze_excitation,
        squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio,
    )


class MultiTalentNexToUEncoder(nn.Module):
    def __init__(self,
                 input_channels: int | dict,
                 patch_size: List[int],
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 return_skips: bool = False,
                 disable_default_stem: bool = False,
                 stem_channels: int = None,
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16):
        super().__init__()
        kernel_sizes = _as_list(kernel_sizes, n_stages)
        features_per_stage = _as_list(features_per_stage, n_stages)
        n_blocks_per_stage = _as_list(n_blocks_per_stage, n_stages)
        strides = _as_list(strides, n_stages)
        bottleneck_channels = _as_list(bottleneck_channels, n_stages)

        img_shape_list, n_size_list = _compute_img_shapes(patch_size, strides, conv_op)
        opt = OptInit(pool_op_kernel_sizes_len=len(strides))
        opt.img_min_shape = img_shape_list[-1]
        opt.n_size_list = n_size_list
        self.opt = opt
        self.n_swin_gnn_stages = 0
        self.no_pool_gnn_stage_num = max(0, n_stages - 4)
        self.n_conv_stages = max(0, self.no_pool_gnn_stage_num - self.n_swin_gnn_stages)

        if not disable_default_stem:
            stem_channels = features_per_stage[0] if stem_channels is None else stem_channels
            if isinstance(input_channels, dict):
                stems = {
                    str(dataset_id): StackedConvBlocks(
                        1, conv_op, int(channels), stem_channels, kernel_sizes[0], 1, conv_bias,
                        norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
                    )
                    for dataset_id, channels in input_channels.items()
                }
                self.stem = nn.ModuleDict(stems)
            else:
                self.stem = StackedConvBlocks(
                    1, conv_op, input_channels, stem_channels, kernel_sizes[0], 1, conv_bias,
                    norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
                )
            input_channels = stem_channels
        else:
            self.stem = None

        stages = []
        for s in range(n_stages):
            stride_for_conv = strides[s]
            common_kwargs = dict(
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=features_per_stage[s],
                kernel_size=kernel_sizes[s],
                initial_stride=stride_for_conv,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                dropout_op=dropout_op,
                dropout_op_kwargs=dropout_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                block=block,
                bottleneck_channels=bottleneck_channels[s],
                stochastic_depth_p=stochastic_depth_p,
                squeeze_excitation=squeeze_excitation,
                squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio,
            )
            if s < self.n_conv_stages:
                stage = _make_residual_blocks(n_blocks_per_stage[s], **common_kwargs)
            elif s < self.no_pool_gnn_stage_num:
                stage = nn.Sequential(
                    _make_residual_blocks(max(1, int(n_blocks_per_stage[s]) - 1), **common_kwargs),
                    SwinGNNBlocks(features_per_stage[s], img_shape_list[s], s - self.n_conv_stages,
                                  opt=self.opt, conv_op=conv_op, norm_op=norm_op,
                                  norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op),
                )
            else:
                stage = nn.Sequential(
                    _make_residual_blocks(max(1, int(n_blocks_per_stage[s]) - 1), **common_kwargs),
                    PoolGNNBlocks(features_per_stage[s], img_shape_list[s], s - self.no_pool_gnn_stage_num,
                                  self.no_pool_gnn_stage_num, opt=self.opt, conv_op=conv_op,
                                  norm_op=norm_op, norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op),
                    SwinGNNBlocks(features_per_stage[s], img_shape_list[s], s - self.n_conv_stages,
                                  opt=self.opt, conv_op=conv_op, norm_op=norm_op,
                                  norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op),
                )
            stages.append(stage)
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x, ids=None):
        if isinstance(ids, str):
            ids = [ids]
        if self.stem is not None:
            x_from_stem = None
            if isinstance(self.stem, nn.ModuleDict):
                for sample, dataset_id in zip(x, ids):
                    sample = self.stem[str(dataset_id)](sample)
                    x_from_stem = sample if x_from_stem is None else torch.cat((x_from_stem, sample), dim=0)
            else:
                for sample in x:
                    sample = self.stem(sample)
                    x_from_stem = sample if x_from_stem is None else torch.cat((x_from_stem, sample), dim=0)
            x = x_from_stem
        else:
            x_cat = None
            for sample in x:
                x_cat = sample if x_cat is None else torch.cat((x_cat, sample), dim=0)
            x = x_cat

        ret = []
        for stage in self.stages:
            x = stage(x)
            ret.append(x)
        return ret if self.return_skips else ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        return np.int64(0)


class MultiTalentNexToUDecoder(nn.Module):
    def __init__(self,
                 encoder: MultiTalentNexToUEncoder,
                 patch_size: List[int],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 num_classes: dict,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.max_num_classes = int(np.max([num_classes[i] for i in num_classes.keys()]))
        n_stages_encoder = len(encoder.output_channels)
        n_conv_per_stage = _as_list(n_conv_per_stage, n_stages_encoder - 1)
        strides = _as_list(strides, n_stages_encoder)
        img_shape_list, n_size_list = _compute_img_shapes(patch_size, strides, encoder.conv_op)

        opt = OptInit(pool_op_kernel_sizes_len=len(strides))
        opt.img_min_shape = img_shape_list[-1]
        opt.n_size_list = n_size_list
        self.opt = opt
        self.n_swin_gnn_stages = 0
        self.no_pool_gnn_stage_num = max(0, n_stages_encoder - 4)
        self.n_conv_stages = max(0, self.no_pool_gnn_stage_num - self.n_swin_gnn_stages)

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        stages = []
        transpconvs = []
        seg_layers = {}
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))

            conv_blocks = _make_residual_blocks(
                max(1, int(n_conv_per_stage[s - 1])),
                encoder.conv_op,
                2 * input_features_skip,
                input_features_skip,
                encoder.kernel_sizes[-(s + 1)],
                1,
                encoder.conv_bias,
                encoder.norm_op,
                encoder.norm_op_kwargs,
                encoder.dropout_op,
                encoder.dropout_op_kwargs,
                encoder.nonlin,
                encoder.nonlin_kwargs,
            )
            if s < (n_stages_encoder - self.no_pool_gnn_stage_num):
                stage = nn.Sequential(
                    conv_blocks,
                    PoolGNNBlocks(input_features_skip, img_shape_list[n_stages_encoder - (s + 1)],
                                  n_stages_encoder - self.no_pool_gnn_stage_num - (s + 1),
                                  self.no_pool_gnn_stage_num, opt=self.opt, conv_op=encoder.conv_op,
                                  norm_op=encoder.norm_op, norm_op_kwargs=encoder.norm_op_kwargs,
                                  dropout_op=encoder.dropout_op),
                    SwinGNNBlocks(input_features_skip, img_shape_list[n_stages_encoder - (s + 1)],
                                  n_stages_encoder - self.n_conv_stages - (s + 1), opt=self.opt,
                                  conv_op=encoder.conv_op, norm_op=encoder.norm_op,
                                  norm_op_kwargs=encoder.norm_op_kwargs, dropout_op=encoder.dropout_op),
                )
            elif s < (n_stages_encoder - self.n_conv_stages):
                stage = nn.Sequential(
                    conv_blocks,
                    SwinGNNBlocks(input_features_skip, img_shape_list[n_stages_encoder - (s + 1)],
                                  n_stages_encoder - self.n_conv_stages - (s + 1), opt=self.opt,
                                  conv_op=encoder.conv_op, norm_op=encoder.norm_op,
                                  norm_op_kwargs=encoder.norm_op_kwargs, dropout_op=encoder.dropout_op),
                )
            else:
                stage = conv_blocks
            stages.append(stage)

            for dataset_id in num_classes.keys():
                dataset_id = str(dataset_id)
                seg_layers.setdefault(dataset_id, [])
                seg_layers[dataset_id].append(encoder.conv_op(input_features_skip, num_classes[dataset_id], 1, 1, 0,
                                                              bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleDict({k: nn.ModuleList(v) for k, v in seg_layers.items()})
        self.has_final_segmentation_heads = True

    def _normalize_ids(self, ids):
        if isinstance(ids, str):
            ids = [ids]
        return [str(i) for i in ids]

    def _forward_stages(self, skips):
        lres_input = skips[-1]
        stage_features = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
            stage_features.append(x)
            lres_input = x
        return stage_features

    def _segment_stage_features(self, stage_features, ids, stage_indices):
        seg_outputs = [[] for _ in range(stage_features[0].shape[0])]
        for s in stage_indices:
            x = stage_features[s]
            for b in range(x.shape[0]):
                seg_outputs[b].append(self.seg_layers[ids[b]][s](torch.unsqueeze(x[b], dim=0)))

        for b in range(len(seg_outputs)):
            seg_outputs[b] = seg_outputs[b][::-1]
        return seg_outputs

    def remove_final_segmentation_heads(self):
        """Remove static full-resolution heads when a feature-level head replaces them."""
        if not self.has_final_segmentation_heads:
            return
        for dataset_id in self.seg_layers.keys():
            del self.seg_layers[dataset_id][-1]
        self.has_final_segmentation_heads = False

    def forward_features(self, skips, ids):
        """Return the final decoder feature and lower-resolution DS logits.

        The full-resolution prediction is intentionally omitted so a caller can
        apply a method-specific head directly to the final decoder feature.
        """
        ids = self._normalize_ids(ids)
        if len(ids) != skips[0].shape[0]:
            raise ValueError(f"Expected {skips[0].shape[0]} dataset ids, got {len(ids)}")
        stage_features = self._forward_stages(skips)
        if self.deep_supervision:
            auxiliary_outputs = self._segment_stage_features(
                stage_features, ids, range(len(stage_features) - 1)
            )
        else:
            auxiliary_outputs = [[] for _ in range(stage_features[-1].shape[0])]
        return stage_features[-1], auxiliary_outputs

    def forward(self, skips, ids):
        ids = self._normalize_ids(ids)
        if len(ids) != skips[0].shape[0]:
            raise ValueError(f"Expected {skips[0].shape[0]} dataset ids, got {len(ids)}")
        if not self.has_final_segmentation_heads:
            raise RuntimeError(
                "The static final segmentation heads were removed. Use forward_features through the "
                "feature-conditioned wrapper instead of calling the base decoder directly."
            )
        stage_features = self._forward_stages(skips)
        if self.deep_supervision:
            stage_indices = range(len(stage_features))
        else:
            stage_indices = [len(stage_features) - 1]
        seg_outputs = self._segment_stage_features(stage_features, ids, stage_indices)
        return seg_outputs if self.deep_supervision else seg_outputs[0]

    def compute_conv_feature_map_size(self, input_size):
        return np.int64(0)


class MultiTalentNexToU(nn.Module):
    def __init__(self,
                 input_channels: int | dict,
                 patch_size: List[int],
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: dict,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None):
        super().__init__()
        self.encoder = MultiTalentNexToUEncoder(
            input_channels, patch_size, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
            n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
            nonlin, nonlin_kwargs, block, bottleneck_channels, return_skips=True, stem_channels=stem_channels
        )
        self.decoder = MultiTalentNexToUDecoder(
            self.encoder, patch_size, strides, {str(k): int(v) for k, v in num_classes.items()},
            n_conv_per_stage_decoder, deep_supervision
        )

    def forward(self, x, ids):
        skips = self.encoder(x, ids)
        return self.decoder(skips, ids)

    def compute_conv_feature_map_size(self, input_size):
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)
