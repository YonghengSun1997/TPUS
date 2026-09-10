# TPUS modification: adapts CLIP-conditioned dynamic feature heads to MultiTalent multiclass datasets.
# Publication of this CLIP-Driven-derived adaptation is separately authorized; see THIRD_PARTY_NOTICES.md.
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class ClipDrivenFeatureDynamicHead(nn.Module):
    """CLIP-Driven Universal Model dynamic head adapted to MultiTalent classes.

    The method-defining path follows ``model/Universal_model.py`` from the
    CLIP-Driven Universal Model: raw frozen CLIP text features and a GAP-pooled
    bottleneck feature are projected to 256 channels, concatenated, and passed
    through a shared controller. The generated parameters operate on an
    8-channel projection of the final decoder feature, never on segmentation
    logits.
    """

    def __init__(
            self,
            num_classes_by_id: dict[str, int],
            bottleneck_channels: int,
            decoder_channels: int,
            conv_op: type[nn.Module],
            clip_embedding_dim: int = 512,
            conditioning_dim: int = 256,
            dynamic_channels: int = 8,
            group_norm_groups: int = 16,
            clip_model_name: str | None = None,
            clip_prompt_template: str | None = None,
            clip_prompt_texts_json: str | None = None,
            clip_prompt_cache: str | None = None,
            clip_device: str | None = None,
    ):
        super().__init__()
        if conv_op is not nn.Conv3d:
            raise ValueError("The CLIP-conditioned TPUS feature head currently supports Conv3d only.")

        self.num_classes_by_id = {str(k): int(v) for k, v in num_classes_by_id.items()}
        self.bottleneck_channels = int(bottleneck_channels)
        self.decoder_channels = int(decoder_channels)
        self.clip_embedding_dim = int(clip_embedding_dim)
        self.conditioning_dim = int(conditioning_dim)
        self.dynamic_channels = int(dynamic_channels)
        self.group_norm_groups = int(group_norm_groups)
        self.clip_model_name = str(clip_model_name or os.environ.get("MT_CLIP_PROMPT_MODEL", "ViT-B/32"))
        self.clip_prompt_template = str(
            clip_prompt_template or os.environ.get("MT_CLIP_PROMPT_TEMPLATE", "A medical image of {label}")
        )
        self.clip_prompt_texts_json = str(
            clip_prompt_texts_json or os.environ.get("MT_CLIP_PROMPT_TEXTS_JSON", "")
        ).strip()
        self.clip_prompt_cache = str(
            clip_prompt_cache or os.environ.get("MT_FAITHFUL_CLIP_PROMPT_CACHE", "")
        ).strip()
        self.clip_device = str(clip_device or os.environ.get("MT_CLIP_PROMPT_DEVICE", "cpu")).strip()

        for name, channels in (
            ("bottleneck", self.bottleneck_channels),
            ("final decoder", self.decoder_channels),
        ):
            if channels % self.group_norm_groups != 0:
                raise ValueError(
                    f"{name} channels ({channels}) must be divisible by the reference GroupNorm group count "
                    f"({self.group_norm_groups})."
                )

        self.GAP = nn.Sequential(
            nn.GroupNorm(self.group_norm_groups, self.bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(self.bottleneck_channels, self.conditioning_dim, kernel_size=1),
        )
        self.precls_conv = nn.Sequential(
            nn.GroupNorm(self.group_norm_groups, self.decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.decoder_channels, self.dynamic_channels, kernel_size=1),
        )
        self.text_to_vision = nn.Linear(self.clip_embedding_dim, self.conditioning_dim)

        self.weight_nums = [
            self.dynamic_channels * self.dynamic_channels,
            self.dynamic_channels * self.dynamic_channels,
            self.dynamic_channels,
        ]
        self.bias_nums = [self.dynamic_channels, self.dynamic_channels, 1]
        self.num_dynamic_params = sum(self.weight_nums) + sum(self.bias_nums)
        self.controller = nn.Conv3d(
            self.conditioning_dim * 2,
            self.num_dynamic_params,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self._id_to_module_key = {
            dataset_id: re.sub(r"[^0-9A-Za-z_]", "_", dataset_id)
            for dataset_id in self.num_classes_by_id
        }
        if len(set(self._id_to_module_key.values())) != len(self._id_to_module_key):
            raise ValueError(f"Dataset ids collide after sanitizing: {self._id_to_module_key}")

        self._clip_text_buffer_names: dict[str, str] = {}
        self._clip_embeddings_ready: dict[str, bool] = {}
        for dataset_id, num_classes in self.num_classes_by_id.items():
            buffer_name = f"clip_text_embeddings_{self._id_to_module_key[dataset_id]}"
            self.register_buffer(
                buffer_name,
                torch.zeros(num_classes, self.clip_embedding_dim, dtype=torch.float32),
                persistent=True,
            )
            self._clip_text_buffer_names[dataset_id] = buffer_name
            self._clip_embeddings_ready[dataset_id] = False

        self.prompt_texts_by_id: dict[str, list[str]] = {
            dataset_id: [f"class_{i}" for i in range(num_classes)]
            for dataset_id, num_classes in self.num_classes_by_id.items()
        }

    @staticmethod
    def _label_sort_value(value: Any) -> int:
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, (list, tuple)):
            numeric = [int(i) for i in value if isinstance(i, (int, float))]
            return min(numeric) if numeric else 0
        return 0

    def _format_prompt_text(self, label_name: str, dataset_id: str, label_index: int) -> str:
        try:
            return self.clip_prompt_template.format(
                label=str(label_name).replace("_", " "),
                dataset_id=str(dataset_id),
                index=int(label_index),
            )
        except KeyError as exc:
            raise KeyError(
                "MT_CLIP_PROMPT_TEMPLATE supports only {label}, {dataset_id}, and {index}."
            ) from exc

    def _load_prompt_text_overrides(self) -> dict[str, list[str]]:
        if not self.clip_prompt_texts_json:
            return {}
        path = Path(self.clip_prompt_texts_json).expanduser()
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "prompt_texts_by_id" in payload:
            payload = payload["prompt_texts_by_id"]
        if not isinstance(payload, dict):
            raise ValueError("MT_CLIP_PROMPT_TEXTS_JSON must map dataset ids to prompt text lists.")
        return {str(dataset_id): [str(text) for text in texts] for dataset_id, texts in payload.items()}

    def update_prompt_texts_from_dataset_jsons(self, dataset_jsons: dict[str, dict]) -> None:
        dataset_jsons = {str(dataset_id): value for dataset_id, value in dataset_jsons.items()}
        overrides = self._load_prompt_text_overrides()
        for dataset_id, num_classes in self.num_classes_by_id.items():
            dataset_json = dataset_jsons.get(dataset_id, {})
            labels = dataset_json.get("labels", {})
            ordered_labels = sorted(labels.items(), key=lambda item: self._label_sort_value(item[1]))
            prompt_texts = [
                self._format_prompt_text(label_name, dataset_id, index)
                for index, (label_name, _) in enumerate(ordered_labels)
            ]
            if dataset_id in overrides:
                prompt_texts = overrides[dataset_id]
            if len(prompt_texts) != num_classes:
                raise ValueError(
                    f"Dataset {dataset_id} requires exactly {num_classes} prompt texts, got {len(prompt_texts)}."
                )
            self.prompt_texts_by_id[dataset_id] = list(prompt_texts)
            self._refresh_clip_text_embeddings(dataset_id, prompt_texts)

    @staticmethod
    def _torch_load(path: Path):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _load_cached_clip_embeddings(self, dataset_id: str, prompt_texts: list[str]) -> torch.Tensor | None:
        if not self.clip_prompt_cache:
            return None
        cache_path = Path(self.clip_prompt_cache).expanduser()
        if not cache_path.is_file():
            return None
        payload = self._torch_load(cache_path)
        if not isinstance(payload, dict):
            return None
        entries = payload.get("embeddings_by_id", payload)
        if not isinstance(entries, dict):
            return None
        entry = entries.get(dataset_id)
        if not isinstance(entry, dict):
            return None
        if entry.get("clip_model_name", self.clip_model_name) != self.clip_model_name:
            return None
        if entry.get("prompt_texts") != prompt_texts:
            return None
        if entry.get("normalized") is not False:
            return None
        embeddings = torch.as_tensor(entry.get("embeddings"), dtype=torch.float32)
        return embeddings

    def _encode_clip_prompt_texts(self, prompt_texts: list[str]) -> torch.Tensor:
        try:
            import clip
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Raw CLIP prompt encoding requires OpenAI CLIP or a cache created with normalized=false."
            ) from exc

        model, _ = clip.load(self.clip_model_name, device=self.clip_device)
        model.eval()
        tokens = clip.tokenize(prompt_texts).to(self.clip_device)
        with torch.no_grad():
            # The reference implementation stores encode_text output directly.
            embeddings = model.encode_text(tokens).float()
        return embeddings.cpu()

    def _save_cached_clip_embeddings(
            self, dataset_id: str, prompt_texts: list[str], embeddings: torch.Tensor
    ) -> None:
        if not self.clip_prompt_cache:
            return
        cache_path = Path(self.clip_prompt_cache).expanduser()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"embeddings_by_id": {}}
        if cache_path.is_file():
            existing = self._torch_load(cache_path)
            if isinstance(existing, dict):
                payload = existing
        entries = payload.setdefault("embeddings_by_id", {})
        entries[dataset_id] = {
            "clip_model_name": self.clip_model_name,
            "prompt_texts": list(prompt_texts),
            "normalized": False,
            "embeddings": embeddings.detach().cpu(),
        }
        temp_path = cache_path.with_suffix(f"{cache_path.suffix}.{os.getpid()}.tmp")
        torch.save(payload, temp_path)
        os.replace(temp_path, cache_path)

    def _refresh_clip_text_embeddings(self, dataset_id: str, prompt_texts: list[str]) -> None:
        embeddings = self._load_cached_clip_embeddings(dataset_id, prompt_texts)
        if embeddings is None:
            embeddings = self._encode_clip_prompt_texts(prompt_texts)
            self._save_cached_clip_embeddings(dataset_id, prompt_texts, embeddings)
        buffer = getattr(self, self._clip_text_buffer_names[dataset_id])
        if tuple(embeddings.shape) != tuple(buffer.shape):
            raise ValueError(
                f"CLIP embeddings for {dataset_id} have shape {tuple(embeddings.shape)}, "
                f"expected {tuple(buffer.shape)}."
            )
        with torch.no_grad():
            buffer.copy_(embeddings.to(device=buffer.device, dtype=buffer.dtype))
        self._clip_embeddings_ready[dataset_id] = True

    def _get_task_encoding(self, dataset_id: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self._clip_embeddings_ready.get(dataset_id, False):
            buffer = getattr(self, self._clip_text_buffer_names[dataset_id])
            if torch.count_nonzero(buffer.detach()).item():
                self._clip_embeddings_ready[dataset_id] = True
        if not self._clip_embeddings_ready.get(dataset_id, False):
            raise RuntimeError(
                f"CLIP embeddings for dataset {dataset_id} are not initialized; call "
                "update_prompt_texts_from_dataset_jsons first."
            )
        buffer = getattr(self, self._clip_text_buffer_names[dataset_id])
        return F.relu(self.text_to_vision(buffer.to(device=device, dtype=dtype)))

    def parse_dynamic_params(self, params: torch.Tensor):
        if params.dim() != 2 or params.size(1) != self.num_dynamic_params:
            raise ValueError(
                f"Expected dynamic params [N, {self.num_dynamic_params}], got {tuple(params.shape)}."
            )
        num_instances = params.size(0)
        splits = torch.split(params, self.weight_nums + self.bias_nums, dim=1)
        weights = list(splits[:len(self.weight_nums)])
        biases = list(splits[len(self.weight_nums):])
        weights[0] = weights[0].reshape(
            num_instances * self.dynamic_channels, self.dynamic_channels, 1, 1, 1
        )
        biases[0] = biases[0].reshape(num_instances * self.dynamic_channels)
        weights[1] = weights[1].reshape(
            num_instances * self.dynamic_channels, self.dynamic_channels, 1, 1, 1
        )
        biases[1] = biases[1].reshape(num_instances * self.dynamic_channels)
        weights[2] = weights[2].reshape(num_instances, self.dynamic_channels, 1, 1, 1)
        biases[2] = biases[2].reshape(num_instances)
        return weights, biases

    @staticmethod
    def heads_forward(features, weights, biases, num_instances):
        x = features
        for index, (weight, bias) in enumerate(zip(weights, biases)):
            x = F.conv3d(x, weight, bias=bias, stride=1, padding=0, groups=num_instances)
            if index < len(weights) - 1:
                x = F.relu(x)
        return x

    def forward(
            self,
            bottleneck: torch.Tensor,
            final_decoder_feature: torch.Tensor,
            ids: list[str],
    ) -> list[torch.Tensor]:
        ids = [str(dataset_id) for dataset_id in ids]
        batch_size = bottleneck.shape[0]
        if final_decoder_feature.shape[0] != batch_size or len(ids) != batch_size:
            raise ValueError(
                "Bottleneck, final decoder feature, and dataset ids must have the same batch size."
            )
        if bottleneck.shape[1] != self.bottleneck_channels:
            raise ValueError(
                f"Expected {self.bottleneck_channels} bottleneck channels, got {bottleneck.shape[1]}."
            )
        if final_decoder_feature.shape[1] != self.decoder_channels:
            raise ValueError(
                f"Expected {self.decoder_channels} decoder channels, got {final_decoder_feature.shape[1]}."
            )

        image_encoding = self.GAP(bottleneck)
        head_features = self.precls_conv(final_decoder_feature)
        outputs = []
        for batch_index, dataset_id in enumerate(ids):
            if dataset_id not in self.num_classes_by_id:
                raise KeyError(f"No dynamic head prompts are configured for dataset {dataset_id}.")
            class_num = self.num_classes_by_id[dataset_id]
            task_encoding = self._get_task_encoding(
                dataset_id, image_encoding.device, image_encoding.dtype
            ).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            image_condition = image_encoding[batch_index:batch_index + 1].repeat(class_num, 1, 1, 1, 1)
            controller_input = torch.cat([image_condition, task_encoding], dim=1)
            params = self.controller(controller_input).flatten(1)

            sample_features = head_features[batch_index:batch_index + 1].repeat(class_num, 1, 1, 1, 1)
            spatial_shape = sample_features.shape[2:]
            sample_features = sample_features.reshape(1, class_num * self.dynamic_channels, *spatial_shape)
            weights, biases = self.parse_dynamic_params(params)
            logits = self.heads_forward(sample_features, weights, biases, class_num)
            outputs.append(logits.reshape(1, class_num, *spatial_shape))
        return outputs

    def get_metadata(self) -> dict[str, Any]:
        return {
            "method": "CLIP-Driven Universal Model feature-conditioned dynamic head",
            "reference_file": "CLIP-Driven-Universal-Model-main/model/Universal_model.py",
            "num_classes_by_id": self.num_classes_by_id,
            "bottleneck_channels": self.bottleneck_channels,
            "decoder_channels": self.decoder_channels,
            "clip_embedding_dim": self.clip_embedding_dim,
            "conditioning_dim": self.conditioning_dim,
            "dynamic_channels": self.dynamic_channels,
            "dynamic_path": f"{self.decoder_channels}->{self.dynamic_channels}->"
                            f"{self.dynamic_channels}->{self.dynamic_channels}->1 per class",
            "num_dynamic_params_per_class": self.num_dynamic_params,
            "clip_model_name": self.clip_model_name,
            "clip_prompt_texts_json": self.clip_prompt_texts_json or None,
            "clip_prompt_cache": self.clip_prompt_cache or None,
            "clip_embeddings_normalized": False,
            "clip_embeddings_ready": dict(self._clip_embeddings_ready),
            "prompt_texts_by_id": self.prompt_texts_by_id,
            "controller_shared_across_datasets_and_classes": True,
            "operates_on_logits": False,
            "residual_logit_fusion": False,
        }


class ClipDrivenFeatureMultiTalentWrapper(nn.Module):
    """Run MultiTalent with a CLIP-conditioned head on final decoder features."""

    def __init__(self, base_network: nn.Module, num_classes_by_id: dict[str, int]):
        super().__init__()
        self.base_network = base_network
        self.num_classes_by_id = {str(k): int(v) for k, v in num_classes_by_id.items()}

        encoder = getattr(base_network, "encoder", None)
        decoder = getattr(base_network, "decoder", None)
        if encoder is None or decoder is None:
            raise ValueError("The feature-conditioned wrapper requires encoder and decoder attributes.")
        if not hasattr(decoder, "forward_features") or not hasattr(decoder, "remove_final_segmentation_heads"):
            raise ValueError("The decoder does not expose the required feature-level interface.")

        self.dynamic_head = ClipDrivenFeatureDynamicHead(
            self.num_classes_by_id,
            bottleneck_channels=int(encoder.output_channels[-1]),
            decoder_channels=int(encoder.output_channels[0]),
            conv_op=encoder.conv_op,
        )
        decoder.remove_final_segmentation_heads()

    @property
    def encoder(self) -> nn.Module:
        return self.base_network.encoder

    @property
    def decoder(self) -> nn.Module:
        return self.base_network.decoder

    @staticmethod
    def _normalize_ids(ids: str | list[str] | tuple[str, ...]) -> list[str]:
        if isinstance(ids, str):
            return [ids]
        return [str(dataset_id) for dataset_id in ids]

    def update_prompt_texts_from_dataset_jsons(self, dataset_jsons: dict[str, dict]) -> None:
        self.dynamic_head.update_prompt_texts_from_dataset_jsons(dataset_jsons)

    def forward(self, x, ids):
        ids = self._normalize_ids(ids)
        skips = list(self.encoder(x, ids))
        bottleneck = skips[-1]
        final_decoder_feature, auxiliary_outputs = self.decoder.forward_features(skips, ids)
        primary_outputs = self.dynamic_head(bottleneck, final_decoder_feature, ids)

        per_sample_outputs = [
            [primary_outputs[index], *auxiliary_outputs[index]]
            for index in range(len(primary_outputs))
        ]
        if self.decoder.deep_supervision:
            return per_sample_outputs
        return [outputs[0] for outputs in per_sample_outputs]

    def compute_conv_feature_map_size(self, input_size):
        return self.base_network.compute_conv_feature_map_size(input_size)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        has_wrapper_keys = any(key.startswith("base_network.") for key in state_dict.keys())
        if has_wrapper_keys:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        own_keys = self.state_dict().keys()
        remapped = {}
        for key, value in state_dict.items():
            base_key = f"base_network.{key}"
            remapped[base_key if base_key in own_keys else key] = value
        return super().load_state_dict(remapped, strict=False, assign=assign)

    def get_ablation_metadata(self) -> dict[str, Any]:
        stems = getattr(self.encoder, "stem", None)
        return {
            "base_network": self.base_network.__class__.__name__,
            "stem_mode": "dataset_specific" if isinstance(stems, nn.ModuleDict) else "shared",
            "stem_dataset_ids": sorted(stems.keys()) if isinstance(stems, nn.ModuleDict) else None,
            "fixed_full_resolution_segmentation_heads_removed": (
                not self.decoder.has_final_segmentation_heads
            ),
            "deep_supervision": bool(self.decoder.deep_supervision),
            "primary_output": "dynamic head on final decoder feature",
            "auxiliary_outputs": "static dataset-specific heads on lower decoder stages",
            "dynamic_head": self.dynamic_head.get_metadata(),
        }
