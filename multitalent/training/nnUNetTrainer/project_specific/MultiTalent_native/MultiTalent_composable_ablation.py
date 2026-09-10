# TPUS modification: adds composable MultiTalent prompt, topology, GNN, and dynamic-head ablations.
from __future__ import annotations

import pydoc
from copy import deepcopy
from typing import List, Tuple, Union

import torch
from batchgenerators.utilities.file_and_folder_operations import join, save_json
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from torch import nn
from torch._dynamo import OptimizedModule

from multitalent.network_architecture.nextou.multitalent_nextou import MultiTalentNexToU
from multitalent.training.nnUNetTrainer.project_specific.MultiTalent_native.MultiTalent_nextou_ablation import (
    _NexToUTopologyLossMixin,
)
from multitalent.training.nnUNetTrainer.project_specific.MultiTalent_native.MultiTalent_trainer import (
    MultiTalent_trainer,
)
from multitalent.utilities.MultiTalent.ablation_feature_wrappers import MultiTalentAblationWrapper
from multitalent.utilities.network_initialization import InitWeights_He


def _build_nextou_gnn_network(
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int | dict,
        num_output_channels: dict,
        enable_deep_supervision: bool = True,
) -> nn.Module:
    architecture_kwargs = deepcopy(arch_init_kwargs)
    for ri in arch_init_kwargs_req_import:
        if architecture_kwargs.get(ri) is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    patch_size = architecture_kwargs.pop("patch_size", None)
    if patch_size is None:
        raise RuntimeError(
            "NexToU GNN ablations require patch_size in architecture arch_kwargs. "
            "Train with one of the MultiTalent_trainer_ab_*_gnn trainers first so it can save patched plans."
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


class MultiTalent_trainer_composable_ablation(_NexToUTopologyLossMixin, MultiTalent_trainer):
    """Base class for independent MultiTalent ablation feature switches."""

    prompt_style: str = "none"
    topology_loss_kind: str | None = None
    use_nextou_gnn: bool = False
    enable_dynamic_conv: bool = False

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        if self.use_nextou_gnn:
            self._inject_patch_size_into_arch_kwargs()

    @property
    def enable_clip_prompt(self) -> bool:
        return self.prompt_style == "clip"

    @property
    def enable_uniseg_prompt(self) -> bool:
        return self.prompt_style == "uniseg"

    def _inject_patch_size_into_arch_kwargs(self) -> None:
        patch_size = list(self.configuration_manager.patch_size)
        self.configuration_manager.configuration['architecture']['arch_kwargs']['patch_size'] = patch_size
        self.plans_manager.plans['configurations'][self.configuration_name]['architecture']['arch_kwargs'][
            'patch_size'] = patch_size
        for dataset_id in self.all_ids:
            self.configuration_managers[dataset_id].configuration['architecture']['arch_kwargs'][
                'patch_size'] = patch_size
            self.plans_managers[dataset_id].plans['configurations'][self.configuration_name]['architecture'][
                'arch_kwargs']['patch_size'] = patch_size

    def _build_loss(self, id):
        if self.topology_loss_kind is None:
            return MultiTalent_trainer._build_loss(self, id)
        return _NexToUTopologyLossMixin._build_loss(self, id)

    def initialize(self):
        super().initialize()
        network = self._unwrap_network(self.network)
        if hasattr(network, "update_prompt_texts_from_dataset_jsons"):
            network.update_prompt_texts_from_dataset_jsons(self.dataset_jsons)

        if self.local_rank == 0:
            metadata = self.get_ablation_metadata()
            if hasattr(network, "get_ablation_metadata"):
                metadata["network_adapter"] = network.get_ablation_metadata()
            save_json(metadata, join(self.output_folder, "ablation_metadata.json"), sort_keys=False)
            self.print_to_log_file("Ablation metadata saved to ablation_metadata.json")

    @staticmethod
    def _unwrap_network(network: nn.Module) -> nn.Module:
        if isinstance(network, OptimizedModule):
            network = network._orig_mod
        if hasattr(network, "module"):
            network = network.module
            if isinstance(network, OptimizedModule):
                network = network._orig_mod
        return network

    def get_ablation_metadata(self) -> dict:
        return {
            "trainer_class": self.__class__.__name__,
            "prompt_style": self.prompt_style,
            "topology_loss_kind": self.topology_loss_kind,
            "use_nextou_gnn": self.use_nextou_gnn,
            "enable_dynamic_conv": self.enable_dynamic_conv,
            "base_model": "NexToU Pool/Swin GNN MultiTalent" if self.use_nextou_gnn else "original MultiTalent",
            "topology_weight_env": "MT_NEXTOU_TOPOLOGY_WEIGHT",
            "references": {
                "clip_dynamic_conv": "https://github.com/ljwztc/CLIP-Driven-Universal-Model",
                "uniseg_prompt": "https://github.com/yeerwen/UniSeg",
                "nextou": "https://github.com/PengchengShi1220/NexToU",
            },
        }

    def plot_network_architecture(self):
        if self.use_nextou_gnn or self.enable_clip_prompt or self.enable_uniseg_prompt or self.enable_dynamic_conv:
            self.print_to_log_file(
                f"Skipping hiddenlayer architecture plot for ablation trainer {self.__class__.__name__}."
            )
            return
        return super().plot_network_architecture()

    @classmethod
    def build_network_architecture(cls,
                                   architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int | dict,
                                   num_output_channels: dict,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        if cls.use_nextou_gnn:
            base_network = _build_nextou_gnn_network(
                arch_init_kwargs,
                arch_init_kwargs_req_import,
                num_input_channels,
                num_output_channels,
                enable_deep_supervision,
            )
        else:
            base_network = MultiTalent_trainer.build_network_architecture(
                architecture_class_name,
                arch_init_kwargs,
                arch_init_kwargs_req_import,
                num_input_channels,
                num_output_channels,
                enable_deep_supervision,
            )

        if cls.prompt_style == "none" and not cls.enable_dynamic_conv:
            return base_network

        return MultiTalentAblationWrapper(
            base_network,
            num_output_channels,
            enable_clip_prompt=cls.prompt_style == "clip",
            enable_uniseg_prompt=cls.prompt_style == "uniseg",
            enable_dynamic_conv=cls.enable_dynamic_conv,
        )


def _make_ablation_trainer(prompt_style: str, topology_loss_kind: str | None, use_nextou_gnn: bool,
                           enable_dynamic_conv: bool) -> str:
    parts = []
    if prompt_style == "clip":
        parts.append("clip")
    elif prompt_style == "uniseg":
        parts.append("uniseg")
    elif prompt_style != "none":
        raise ValueError(f"Unknown prompt style: {prompt_style}")

    if topology_loss_kind is not None:
        parts.append(topology_loss_kind)
    if use_nextou_gnn:
        parts.append("gnn")
    if enable_dynamic_conv:
        parts.append("dynconv")
    suffix = "_".join(parts) if parts else "base"
    class_name = f"MultiTalent_trainer_ab_{suffix}"
    globals()[class_name] = type(
        class_name,
        (MultiTalent_trainer_composable_ablation,),
        {
            "__module__": __name__,
            "__doc__": (
                "Composable MultiTalent ablation: "
                f"prompt_style={prompt_style}, topology_loss_kind={topology_loss_kind}, "
                f"use_nextou_gnn={use_nextou_gnn}, enable_dynamic_conv={enable_dynamic_conv}."
            ),
            "prompt_style": prompt_style,
            "topology_loss_kind": topology_loss_kind,
            "use_nextou_gnn": use_nextou_gnn,
            "enable_dynamic_conv": enable_dynamic_conv,
        },
    )
    return class_name


_ABLATION_TRAINER_NAMES = []
for _prompt_style in ("none", "clip", "uniseg"):
    for _topology_loss_kind in (None, "ti", "bti"):
        for _use_nextou_gnn in (False, True):
            for _enable_dynamic_conv in (False, True):
                _ABLATION_TRAINER_NAMES.append(
                    _make_ablation_trainer(
                        _prompt_style,
                        _topology_loss_kind,
                        _use_nextou_gnn,
                        _enable_dynamic_conv,
                    )
                )


class MultiTalent_trainer_uniseg_prompt(globals()["MultiTalent_trainer_ab_uniseg"]):
    """Original MultiTalent network with UniSeg-like bottleneck prompt only.

    This is a readability alias for the pure UniSeg prompt ablation. It does
    not enable CLIP-style prompt, CLIP-style dynamic convolution, NexToU
    topology loss, or NexToU GNN blocks.
    """

    prompt_style = "uniseg"
    topology_loss_kind = None
    use_nextou_gnn = False
    enable_dynamic_conv = False


class MultiTalent_trainer_ab_dynconv_fixed(globals()["MultiTalent_trainer_ab_dynconv"]):
    """Stable rerun alias for the dynamic-conv-only ablation.

    This keeps the same feature switches as MultiTalent_trainer_ab_dynconv but
    writes to a separate output folder after the dynamic branch stability patch.
    """

    prompt_style = "none"
    topology_loss_kind = None
    use_nextou_gnn = False
    enable_dynamic_conv = True


class MultiTalent_trainer_ab_gnn_dynconv_fixed(globals()["MultiTalent_trainer_ab_gnn_dynconv"]):
    """Stable rerun alias for the NexToU-GNN + dynamic-conv ablation."""

    prompt_style = "none"
    topology_loss_kind = None
    use_nextou_gnn = True
    enable_dynamic_conv = True


class MultiTalent_trainer_ab_clip_ti_gnn_dynconv_debug1step(
        globals()["MultiTalent_trainer_ab_clip_ti_gnn_dynconv"]):
    """One-train-step debug alias for interactive VS Code inspection."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1
        self.num_iterations_per_epoch = 1
        self.num_val_iterations_per_epoch = 1

    def perform_actual_validation(self, *args, **kwargs):
        self.print_to_log_file("Skipping full-dataset validation in debug1step trainer.")


__all__ = [
    "MultiTalent_trainer_composable_ablation",
    "MultiTalent_trainer_uniseg_prompt",
    "MultiTalent_trainer_ab_dynconv_fixed",
    "MultiTalent_trainer_ab_gnn_dynconv_fixed",
    "MultiTalent_trainer_ab_clip_ti_gnn_dynconv_debug1step",
    *_ABLATION_TRAINER_NAMES,
]
