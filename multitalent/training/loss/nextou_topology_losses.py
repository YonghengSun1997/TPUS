# TPUS modification: adapts NexToU/Topology Interaction losses to MultiTalent label managers.
from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn

from multitalent.training.loss.dice import MemoryEfficientSoftDiceLoss, SoftDiceLoss
from multitalent.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from multitalent.utilities.helpers import softmax_helper_dim1


def make_pairwise_ti_exclusions(num_segmentation_heads: int) -> list[list[int]]:
    foreground_labels = range(1, int(num_segmentation_heads))
    return [list(pair) for pair in combinations(foreground_labels, 2)]


def make_pairwise_bti_exclusions(num_segmentation_heads: int) -> list[list[list[int]]]:
    foreground_labels = range(1, int(num_segmentation_heads))
    return [[[a], [b]] for a, b in combinations(foreground_labels, 2)]


class _BaseTopologyInteractionLoss(nn.Module):
    def __init__(self, dim=3, connectivity=26, inclusion=None, exclusion=None, min_thick=1, binary_groups=False):
        super().__init__()
        self.dim = int(dim)
        self.connectivity = int(connectivity)
        self.min_thick = int(min_thick)
        self.binary_groups = bool(binary_groups)
        self.interaction_list = []
        self.sum_dim_list = None
        self.conv_op = None
        self.apply_nonlin = softmax_helper_dim1
        self.ce_loss_func = nn.CrossEntropyLoss(reduction='none')

        if self.dim == 2:
            self.sum_dim_list = [1, 2, 3]
            self.conv_op = torch.nn.functional.conv2d
        elif self.dim == 3:
            self.sum_dim_list = [1, 2, 3, 4]
            self.conv_op = torch.nn.functional.conv3d
        else:
            raise ValueError(f"Topology loss only supports 2D/3D, got dim={dim}")

        self.register_buffer("kernel", self._make_kernel().double(), persistent=False)

        inclusion = [] if inclusion is None else inclusion
        exclusion = [] if exclusion is None else exclusion
        for inc in inclusion:
            self.interaction_list.append((True, self._as_label_tensor(inc[0]), self._as_label_tensor(inc[1])))
        for exc in exclusion:
            self.interaction_list.append((False, self._as_label_tensor(exc[0]), self._as_label_tensor(exc[1])))

    def _make_kernel(self) -> torch.Tensor:
        k = 2 * self.min_thick + 1
        if self.dim == 2:
            if self.connectivity == 4:
                np_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            elif self.connectivity == 8:
                np_kernel = np.ones((k, k))
            else:
                raise ValueError(f"2D topology connectivity must be 4 or 8, got {self.connectivity}")
        else:
            if self.connectivity == 6:
                np_kernel = np.array([
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                ])
            elif self.connectivity == 26:
                np_kernel = np.ones((k, k, k))
            else:
                raise ValueError(f"3D topology connectivity must be 6 or 26, got {self.connectivity}")
        return torch.from_numpy(np.expand_dims(np.expand_dims(np_kernel, axis=0), axis=0))

    def _as_label_tensor(self, labels) -> torch.Tensor:
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().long().flatten().tolist()
        elif isinstance(labels, (int, np.integer)):
            labels = [int(labels)]
        elif isinstance(labels, Iterable):
            labels = [int(i) for i in labels]
        else:
            labels = [int(labels)]
        return torch.tensor(labels, dtype=torch.long)

    def _mask_for_labels(self, p: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.to(device=p.device)
        if self.binary_groups or labels.numel() > 1:
            return torch.isin(p.long(), labels).double()
        return torch.where(p.long() == labels[0], 1.0, 0.0).double()

    def _critical_voxels(self, prediction: torch.Tensor) -> torch.Tensor:
        critical = None
        kernel = self.kernel.to(device=prediction.device)
        for interaction_type, label_a, label_c in self.interaction_list:
            mask_a = self._mask_for_labels(prediction, label_a)
            if interaction_type:
                mask_c = self._mask_for_labels(prediction, label_c)
                mask_c = torch.logical_not(torch.logical_or(mask_c.bool(), mask_a.bool())).double()
            else:
                mask_c = self._mask_for_labels(prediction, label_c)

            neighbourhood_c = self.conv_op(mask_c, kernel, padding='same')
            neighbourhood_c = torch.where(neighbourhood_c >= 1.0, 1.0, 0.0)
            neighbourhood_a = self.conv_op(mask_a, kernel, padding='same')
            neighbourhood_a = torch.where(neighbourhood_a >= 1.0, 1.0, 0.0)
            violating = torch.where(neighbourhood_c * mask_a + neighbourhood_a * mask_c >= 1.0, 1.0, 0.0)
            critical = violating if critical is None else torch.logical_or(critical.bool(), violating.bool()).double()

        if critical is None:
            return torch.zeros_like(prediction, dtype=torch.double)
        return critical

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_softmax = self.apply_nonlin(x)
        prediction = torch.argmax(x_softmax, dim=1, keepdim=True).double()
        del x_softmax

        critical_voxels = self._critical_voxels(prediction)
        ce_tensor = torch.unsqueeze(self.ce_loss_func(x.double(), y[:, 0].long()), dim=1)
        ce_tensor[:, 0] = ce_tensor[:, 0] * torch.squeeze(critical_voxels, dim=1)
        return ce_tensor.sum(dim=self.sum_dim_list).mean()


class TI_Loss(_BaseTopologyInteractionLoss):
    def __init__(self, dim=3, connectivity=26, inclusion=None, exclusion=None, min_thick=1):
        super().__init__(dim, connectivity, inclusion, exclusion, min_thick, binary_groups=False)


class BTI_Loss(_BaseTopologyInteractionLoss):
    def __init__(self, dim=3, connectivity=26, inclusion=None, exclusion=None, min_thick=1):
        super().__init__(dim, connectivity, inclusion, exclusion, min_thick, binary_groups=True)


class DC_CE_Topology_Loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, topology_kwargs, topology_loss_class,
                 weight_ce=1, weight_dice=1, weight_topology=1e-6, ignore_label=None,
                 dice_class=SoftDiceLoss):
        super().__init__()
        ce_kwargs = dict(ce_kwargs)
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_topology = weight_topology
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.topology = topology_loss_class(**topology_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            if target.shape[1] != 1:
                raise RuntimeError("ignore_label is only implemented for single-channel label maps")
            mask = (target != self.ignore_label).bool()
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
            num_fg = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        topology_loss = self.topology(net_output, target) if self.weight_topology != 0 else 0
        return self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_topology * topology_loss


class DC_and_CE_and_TI_Loss(DC_CE_Topology_Loss):
    def __init__(self, soft_dice_kwargs, ce_kwargs, ti_kwargs, weight_ce=1, weight_dice=1, weight_ti=1e-6,
                 ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):
        super().__init__(soft_dice_kwargs, ce_kwargs, ti_kwargs, TI_Loss, weight_ce, weight_dice, weight_ti,
                         ignore_label, dice_class)


class DC_and_CE_and_BTI_Loss(DC_CE_Topology_Loss):
    def __init__(self, soft_dice_kwargs, ce_kwargs, bti_kwargs, weight_ce=1, weight_dice=1, weight_ti=1e-6,
                 ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):
        super().__init__(soft_dice_kwargs, ce_kwargs, bti_kwargs, BTI_Loss, weight_ce, weight_dice, weight_ti,
                         ignore_label, dice_class)
