# TPUS modification: registers the prompt/dynamic MultiTalent ablation trainers.
from __future__ import annotations

from typing import List, Tuple, Union

import torch
from batchgenerators.utilities.file_and_folder_operations import join, save_json
from torch import nn
from torch._dynamo import OptimizedModule

from multitalent.training.nnUNetTrainer.project_specific.MultiTalent_native.MultiTalent_trainer import MultiTalent_trainer
from multitalent.utilities.MultiTalent.prompt_dynamic_wrapper import PromptDynamicMultiTalentWrapper


class MultiTalent_trainer_prompt_dynamic(MultiTalent_trainer):
    """MultiTalent with learned label prompts and prompt-conditioned dynamic heads."""

    def initialize(self):
        super().initialize()
        network = self.network
        if isinstance(network, OptimizedModule):
            network = network._orig_mod
        if hasattr(network, "module"):
            network = network.module
            if isinstance(network, OptimizedModule):
                network = network._orig_mod

        if hasattr(network, "update_prompt_texts_from_dataset_jsons"):
            network.update_prompt_texts_from_dataset_jsons(self.dataset_jsons)
            if self.local_rank == 0:
                save_json(network.get_prompt_metadata(), join(self.output_folder, "prompt_dynamic_metadata.json"),
                          sort_keys=False)
                self.print_to_log_file("Prompt/dynamic metadata saved to prompt_dynamic_metadata.json")

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int | dict,
                                   num_output_channels: dict,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        base_network = MultiTalent_trainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )
        return PromptDynamicMultiTalentWrapper(base_network, num_output_channels)


class MultiTalent_trainer_prompt_dynamic_1ep(MultiTalent_trainer_prompt_dynamic):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1
