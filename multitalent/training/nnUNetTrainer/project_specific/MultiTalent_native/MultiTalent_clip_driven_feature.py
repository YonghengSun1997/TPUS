# TPUS modification: integrates multistems, NexToU GNN, TI loss, and a CLIP-conditioned feature head.
from __future__ import annotations

from typing import Any, List, Tuple, Union

import torch
from batchgenerators.utilities.file_and_folder_operations import join, save_json
from torch import nn

from multitalent.training.nnUNetTrainer.project_specific.MultiTalent_native.MultiTalent_composable_ablation import (
    MultiTalent_trainer_composable_ablation,
    _build_nextou_gnn_network,
)
from multitalent.training.nnUNetTrainer.project_specific.MultiTalent_native.MultiTalent_multistem import (
    MultiTalent_trainer_multistems,
)
from multitalent.utilities.MultiTalent.clip_driven_feature_wrapper import (
    ClipDrivenFeatureMultiTalentWrapper,
)


class MultiTalent_trainer_clip_ti_gnn_clipdynfeature_multistems(
        MultiTalent_trainer_composable_ablation):
    """Dataset-specific stems plus a CLIP-conditioned dynamic feature head."""

    prompt_style = "clip"
    topology_loss_kind = "ti"
    use_nextou_gnn = True
    enable_dynamic_conv = True
    stem_mode = "dataset_specific"

    @classmethod
    def build_network_architecture(
            cls,
            architecture_class_name: str,
            arch_init_kwargs: dict,
            arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
            num_input_channels: int | dict,
            num_output_channels: dict,
            enable_deep_supervision: bool = True,
    ) -> nn.Module:
        if not isinstance(num_input_channels, dict):
            raise ValueError(
                "This trainer requires a dataset_id -> input_channels mapping so the stems cannot be shared."
            )
        base_network = _build_nextou_gnn_network(
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )
        return ClipDrivenFeatureMultiTalentWrapper(base_network, num_output_channels)

    def initialize(self):
        # Preserve the complete dataset_id -> channel mapping when constructing
        # the encoder. The composable trainer otherwise reduces it to max(C).
        MultiTalent_trainer_multistems.initialize(self)

        network = self._unwrap_network(self.network)
        network.update_prompt_texts_from_dataset_jsons(self.dataset_jsons)
        self._method_validation = self._validate_method(network)

        if self.local_rank == 0:
            metadata = self.get_ablation_metadata()
            metadata["network_adapter"] = network.get_ablation_metadata()
            save_json(metadata, join(self.output_folder, "ablation_metadata.json"), sort_keys=False)
            self.print_to_log_file(
                "CLIP feature-head metadata saved to ablation_metadata.json; "
                f"stems={self._method_validation['stem_dataset_ids']}"
            )

    def _validate_method(self, network: nn.Module) -> dict[str, Any]:
        if not isinstance(network, ClipDrivenFeatureMultiTalentWrapper):
            raise RuntimeError(
                "Expected ClipDrivenFeatureMultiTalentWrapper, "
                f"found {type(network).__name__}."
            )

        stems = getattr(network.encoder, "stem", None)
        if not isinstance(stems, nn.ModuleDict):
            raise RuntimeError(f"Expected dataset-specific ModuleDict stems, found {type(stems).__name__}.")
        expected_ids = {str(dataset_id) for dataset_id in self.input_channels}
        actual_ids = set(stems.keys())
        if actual_ids != expected_ids:
            raise RuntimeError(
                f"Stem ids do not match datasets: expected {sorted(expected_ids)}, got {sorted(actual_ids)}."
            )

        parameter_ids = {
            dataset_id: {id(parameter) for parameter in stems[dataset_id].parameters()}
            for dataset_id in sorted(actual_ids)
        }
        for left_index, left_id in enumerate(sorted(actual_ids)):
            for right_id in sorted(actual_ids)[left_index + 1:]:
                overlap = parameter_ids[left_id] & parameter_ids[right_id]
                if overlap:
                    raise RuntimeError(f"Stems {left_id} and {right_id} share parameter objects.")

        dynamic_head = network.dynamic_head
        if dynamic_head.num_dynamic_params != 153:
            raise RuntimeError(
                f"Reference 8->8->8->1 head requires 153 parameters, got {dynamic_head.num_dynamic_params}."
            )
        if network.decoder.has_final_segmentation_heads:
            raise RuntimeError("Static full-resolution segmentation heads were not removed.")
        if dynamic_head.num_classes_by_id != {
            str(dataset_id): int(num_classes)
            for dataset_id, num_classes in self.all_num_seg_heads.items()
        }:
            raise RuntimeError("Dynamic class heads do not match MultiTalent segmentation heads.")

        return {
            "stem_dataset_ids": sorted(actual_ids),
            "input_channels_by_dataset": {
                str(dataset_id): int(channels)
                for dataset_id, channels in self.input_channels.items()
            },
            "stem_parameter_counts_by_dataset": {
                dataset_id: sum(parameter.numel() for parameter in stems[dataset_id].parameters())
                for dataset_id in sorted(actual_ids)
            },
            "stems_share_parameter_objects": False,
            "shared_after_stem": True,
            "dynamic_head_input": "final decoder feature after fixed precls projection",
            "dynamic_head_operates_on_logits": False,
            "dynamic_parameters_per_class": dynamic_head.num_dynamic_params,
            "fixed_final_heads_removed": True,
        }

    def get_ablation_metadata(self) -> dict:
        metadata = super().get_ablation_metadata()
        metadata.update({
            "trainer_class": self.__class__.__name__,
            "stem_mode": self.stem_mode,
            "clip_dynamic_implementation": "feature_conditioned_adaptation",
            "conditioning": "concat(GAP(bottleneck)->256, raw_CLIP_text->256)",
            "dynamic_target": "precls(final_decoder_feature): C->8, then 8->8->8->1 per class",
            "primary_head": "dynamic",
            "auxiliary_deep_supervision_heads": "static dataset-specific",
            "class_semantics": "dataset-specific multiclass logits including background",
            "deliberate_reference_adaptation": (
                "Preserves MultiTalent softmax labels: Dataset801 has 2 class logits and Dataset802 has 5, "
                "including background, instead of CLIP-Driven foreground-only sigmoid targets."
            ),
        })
        if hasattr(self, "_method_validation"):
            metadata["method_validation"] = self._method_validation
        return metadata


class MultiTalent_trainer_clip_ti_gnn_clipdynfeature_multistems_1ep(
        MultiTalent_trainer_clip_ti_gnn_clipdynfeature_multistems):
    """One-epoch smoke-test variant with identical architecture and loss."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1


__all__ = [
    "MultiTalent_trainer_clip_ti_gnn_clipdynfeature_multistems",
    "MultiTalent_trainer_clip_ti_gnn_clipdynfeature_multistems_1ep",
]
