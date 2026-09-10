# TPUS modification: integrates NexToU GNN and TI/BTI loss ablations with MultiTalent.
from __future__ import annotations

import os
import pydoc
from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np
import torch
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from torch import nn

from multitalent.network_architecture.nextou.multitalent_nextou import MultiTalentNexToU
from multitalent.training.loss.compound_losses import DC_and_BCE_loss
from multitalent.training.loss.deep_supervision import DeepSupervisionWrapper
from multitalent.training.loss.dice import MemoryEfficientSoftDiceLoss
from multitalent.training.loss.nextou_topology_losses import (
    DC_and_CE_and_BTI_Loss,
    DC_and_CE_and_TI_Loss,
    make_pairwise_bti_exclusions,
    make_pairwise_ti_exclusions,
)
from multitalent.training.nnUNetTrainer.project_specific.MultiTalent_native.MultiTalent_trainer import MultiTalent_trainer
from multitalent.utilities.network_initialization import InitWeights_He


class _NexToUTopologyLossMixin:
    topology_loss_kind = "bti"

    def _get_topology_weight(self, dim: int) -> float:
        default_weight = 1e-6 if dim == 3 else 1e-4
        return float(os.environ.get("MT_NEXTOU_TOPOLOGY_WEIGHT", default_weight))

    def _build_loss(self, id):
        if self.label_managers[id].has_regions:
            self.print_to_log_file(
                f"NexToU topology loss is not enabled for region-based dataset {id}; using the original loss."
            )
            if self.is_ddp:
                batchdice = False
            else:
                batchdice = self.configuration_manager.batch_dice
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': batchdice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_managers[id].ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            if self.is_ddp:
                batchdice = False
            else:
                batchdice = self.configuration_manager.batch_dice

            dim = len(self.configuration_manager.patch_size)
            connectivity = 26 if dim == 3 else 8
            lambda_ti = self._get_topology_weight(dim)
            num_segmentation_heads = int(self.all_num_seg_heads[id])

            if self.topology_loss_kind == "ti":
                topology_kwargs = {
                    "dim": dim,
                    "connectivity": connectivity,
                    "inclusion": [],
                    "exclusion": make_pairwise_ti_exclusions(num_segmentation_heads),
                    "min_thick": 1,
                }
                loss_class = DC_and_CE_and_TI_Loss
            elif self.topology_loss_kind == "bti":
                topology_kwargs = {
                    "dim": dim,
                    "connectivity": connectivity,
                    "inclusion": [],
                    "exclusion": make_pairwise_bti_exclusions(num_segmentation_heads),
                    "min_thick": 1,
                }
                loss_class = DC_and_CE_and_BTI_Loss
            else:
                raise ValueError(f"Unknown topology loss kind: {self.topology_loss_kind}")

            loss = loss_class(
                {'batch_dice': batchdice, 'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp},
                {},
                topology_kwargs,
                weight_ce=1,
                weight_dice=1,
                weight_ti=lambda_ti,
                ignore_label=self.label_managers[id].ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
            self.print_to_log_file(
                f"NexToU {self.topology_loss_kind.upper()} loss for dataset {id}: "
                f"dim={dim}, connectivity={connectivity}, lambda={lambda_ti}, "
                f"num_segmentation_heads={num_segmentation_heads}, "
                f"num_exclusions={len(topology_kwargs['exclusion'])}"
            )

        if self._do_i_compile() and hasattr(loss, "dc"):
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


class MultiTalent_trainer_nextou_ti_loss(_NexToUTopologyLossMixin, MultiTalent_trainer):
    """Original MultiTalent network with NexToU-style TI loss only."""

    topology_loss_kind = "ti"


class MultiTalent_trainer_nextou_bti_loss(_NexToUTopologyLossMixin, MultiTalent_trainer):
    """Original MultiTalent network with NexToU-style BTI loss only."""

    topology_loss_kind = "bti"


class MultiTalent_trainer_nextou_topology_loss(MultiTalent_trainer_nextou_bti_loss):
    """Alias for the main topology-loss ablation, defaulting to BTI."""


class MultiTalent_trainer_nextou_gnn(MultiTalent_trainer):
    """MultiTalent trainer using NexToU Pool/Swin GNN blocks with the original MultiTalent loss."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self._inject_patch_size_into_arch_kwargs()

    def _inject_patch_size_into_arch_kwargs(self) -> None:
        patch_size = list(self.configuration_manager.patch_size)
        self.configuration_manager.configuration['architecture']['arch_kwargs']['patch_size'] = patch_size
        self.plans_manager.plans['configurations'][self.configuration_name]['architecture']['arch_kwargs'][
            'patch_size'] = patch_size
        for id in self.all_ids:
            self.configuration_managers[id].configuration['architecture']['arch_kwargs']['patch_size'] = patch_size
            self.plans_managers[id].plans['configurations'][self.configuration_name]['architecture']['arch_kwargs'][
                'patch_size'] = patch_size

    def plot_network_architecture(self):
        self.print_to_log_file("Skipping hiddenlayer architecture plot for MultiTalent NexToU GNN network.")

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int | dict,
                                   num_output_channels: dict,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        architecture_kwargs = deepcopy(arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs.get(ri) is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

        patch_size = architecture_kwargs.pop("patch_size", None)
        if patch_size is None:
            raise RuntimeError(
                "MultiTalent_trainer_nextou_gnn requires patch_size in architecture arch_kwargs. "
                "Train with this trainer first so it can save patched plans for inference."
            )

        if "n_conv_per_stage" in architecture_kwargs and "n_blocks_per_stage" not in architecture_kwargs:
            architecture_kwargs["n_blocks_per_stage"] = architecture_kwargs.pop("n_conv_per_stage")
        architecture_kwargs["deep_supervision"] = enable_deep_supervision

        network = MultiTalentNexToU(
            input_channels=num_input_channels,
            patch_size=patch_size,
            num_classes={str(k): int(v) for k, v in num_output_channels.items()},
            **architecture_kwargs,
        )
        network.apply(InitWeights_He(1e-2))
        network.apply(init_last_bn_before_add_to_0)
        return network


class MultiTalent_trainer_nextou_gnn_1ep(MultiTalent_trainer_nextou_gnn):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1
