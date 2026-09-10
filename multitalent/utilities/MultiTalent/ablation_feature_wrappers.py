# TPUS modification: adds prompt and dynamic-head wrappers used by retained ablation trainers.
# Publication of CLIP-Driven-derived portions is separately authorized; see THIRD_PARTY_NOTICES.md.
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class UniSegBottleneckPromptAdapter(nn.Module):
    """UniSeg-style task prompt fusion at the encoder bottleneck.

    UniSeg concatenates a task prompt at the bottleneck before decoding. The
    MultiTalent decoders in this codebase expect a fixed bottleneck channel
    count, so this adapter uses the same prompt-bank/fusion idea and injects the
    selected task prompt as a gated residual feature map.
    """

    def __init__(
            self,
            dataset_ids: list[str],
            bottleneck_channels: int,
            conv_op: type[nn.Module],
            hidden_channels: int | None = None,
            initial_gate_logit: float = -6.0,
    ):
        super().__init__()
        self.dataset_ids = [str(i) for i in dataset_ids]
        self.dataset_id_to_index = {dataset_id: idx for idx, dataset_id in enumerate(self.dataset_ids)}
        self.bottleneck_channels = int(bottleneck_channels)
        self.num_tasks = len(self.dataset_ids)
        self.spatial_ndim = 3 if conv_op == nn.Conv3d else 2
        hidden_channels = max(8, self.bottleneck_channels // 4) if hidden_channels is None else int(hidden_channels)

        self.intermedia_prompt = nn.Parameter(torch.randn(1, self.num_tasks, *([1] * self.spatial_ndim)))
        self.fusion_layer = nn.Sequential(
            conv_op(self.bottleneck_channels + self.num_tasks, hidden_channels, 1, 1, 0, bias=True),
            nn.LeakyReLU(inplace=True),
            conv_op(hidden_channels, self.num_tasks, 1, 1, 0, bias=True),
        )
        self.prompt_projector = conv_op(1, self.bottleneck_channels, 1, 1, 0, bias=True)
        self.prompt_gate_logit = nn.Parameter(torch.tensor(float(initial_gate_logit)))

    def forward(self, bottleneck: torch.Tensor, ids: list[str]) -> torch.Tensor:
        if bottleneck.ndim not in (4, 5):
            return bottleneck
        if len(ids) == 1 and bottleneck.shape[0] != 1:
            ids = ids * bottleneck.shape[0]
        if len(ids) != bottleneck.shape[0]:
            raise ValueError(f"Expected {bottleneck.shape[0]} ids, got {len(ids)}")

        prompt_bank = self.intermedia_prompt.to(device=bottleneck.device, dtype=bottleneck.dtype)
        prompt_bank = prompt_bank.expand(bottleneck.shape[0], -1, *bottleneck.shape[2:])
        dynamic_prompt_bank = self.fusion_layer(torch.cat([bottleneck, prompt_bank], dim=1))

        selected_prompt = []
        for batch_idx, dataset_id in enumerate(ids):
            dataset_id = str(dataset_id)
            if dataset_id not in self.dataset_id_to_index:
                raise KeyError(f"No UniSeg prompt was configured for dataset id {dataset_id}")
            task_idx = self.dataset_id_to_index[dataset_id]
            selected_prompt.append(dynamic_prompt_bank[batch_idx:batch_idx + 1, task_idx:task_idx + 1])
        selected_prompt = torch.cat(selected_prompt, dim=0)

        prompt_delta = self.prompt_projector(selected_prompt)
        gate = torch.sigmoid(self.prompt_gate_logit).to(device=bottleneck.device, dtype=bottleneck.dtype)
        return bottleneck + gate * prompt_delta

    def get_metadata(self) -> dict[str, Any]:
        return {
            "dataset_ids": self.dataset_ids,
            "bottleneck_channels": self.bottleneck_channels,
            "num_tasks": self.num_tasks,
            "prompt_gate": float(torch.sigmoid(self.prompt_gate_logit).detach().cpu()),
            "reference": "UniSeg bottleneck intermedia_prompt with task-conditioned fusion",
        }


class ClipDynamicLogitAdapter(nn.Module):
    """CLIP-driven Universal Model style class prompts and dynamic logit heads."""

    def __init__(
            self,
            num_classes_by_id: dict[str, int],
            enable_clip_prompt: bool = False,
            enable_dynamic_conv: bool = False,
            prompt_dim: int = 256,
            dynamic_hidden_channels: int = 8,
            controller_hidden_dim: int = 256,
            initial_gate_logit: float = -8.0,
            controller_param_scale: float = 0.1,
            dynamic_logit_scale: float = 0.25,
            prompt_source: str | None = None,
            clip_model_name: str | None = None,
            clip_prompt_template: str | None = None,
            clip_prompt_texts_json: str | None = None,
            clip_prompt_cache: str | None = None,
            clip_device: str | None = None,
    ):
        super().__init__()
        self.num_classes_by_id = {str(k): int(v) for k, v in num_classes_by_id.items()}
        self.enable_clip_prompt = bool(enable_clip_prompt)
        self.enable_dynamic_conv = bool(enable_dynamic_conv)
        self.prompt_dim = int(prompt_dim)
        self.dynamic_hidden_channels = int(dynamic_hidden_channels)
        self.controller_param_scale = float(controller_param_scale)
        self.dynamic_logit_scale = float(dynamic_logit_scale)
        self.prompt_source = self._normalize_prompt_source(
            prompt_source or os.environ.get("MT_CLIP_PROMPT_SOURCE", "learned")
        )
        self.clip_model_name = str(clip_model_name or os.environ.get("MT_CLIP_PROMPT_MODEL", "ViT-B/32"))
        self.clip_prompt_template = str(
            clip_prompt_template or os.environ.get("MT_CLIP_PROMPT_TEMPLATE", "A medical image of {label}")
        )
        self.clip_prompt_texts_json = str(
            clip_prompt_texts_json or os.environ.get("MT_CLIP_PROMPT_TEXTS_JSON", "")
        ).strip()
        self.clip_prompt_cache = str(clip_prompt_cache or os.environ.get("MT_CLIP_PROMPT_CACHE", "")).strip()
        self.clip_device = str(clip_device or os.environ.get("MT_CLIP_PROMPT_DEVICE", "cpu")).strip()
        self.clip_embedding_dim = 512

        self._id_to_module_key = {
            dataset_id: self._safe_module_key(dataset_id) for dataset_id in self.num_classes_by_id
        }
        if len(set(self._id_to_module_key.values())) != len(self._id_to_module_key):
            raise ValueError(f"Dataset ids collide after sanitizing: {self._id_to_module_key}")

        self.class_embeddings = nn.ModuleDict({
            self._id_to_module_key[dataset_id]: nn.Embedding(num_classes, self.prompt_dim)
            for dataset_id, num_classes in self.num_classes_by_id.items()
        })
        self.context_projector = nn.Sequential(
            nn.Linear(1, self.prompt_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.prompt_dim, self.prompt_dim),
        )

        if self.enable_clip_prompt:
            self.prompt_film = nn.Sequential(
                nn.Linear(self.prompt_dim, controller_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(controller_hidden_dim, 2),
            )
            self.prompt_gate_logit = nn.Parameter(torch.tensor(float(initial_gate_logit)))
        else:
            self.prompt_film = None
            self.register_parameter("prompt_gate_logit", None)

        self.weight_nums = [
            self.dynamic_hidden_channels,
            self.dynamic_hidden_channels * self.dynamic_hidden_channels,
            self.dynamic_hidden_channels,
        ]
        self.bias_nums = [
            self.dynamic_hidden_channels,
            self.dynamic_hidden_channels,
            1,
        ]
        self.num_dynamic_params = sum(self.weight_nums) + sum(self.bias_nums)
        if self.enable_dynamic_conv:
            self.controller = nn.Sequential(
                nn.Linear(self.prompt_dim, controller_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(controller_hidden_dim, self.num_dynamic_params),
            )
            self.dynamic_gate_logit = nn.Parameter(torch.tensor(float(initial_gate_logit)))
        else:
            self.controller = None
            self.register_parameter("dynamic_gate_logit", None)

        self.prompt_texts_by_id: dict[str, list[str]] = {
            dataset_id: [f"class_{i}" for i in range(num_classes)]
            for dataset_id, num_classes in self.num_classes_by_id.items()
        }
        self.clip_text_to_prompt = (
            nn.Linear(self.clip_embedding_dim, self.prompt_dim) if self.prompt_source == "clip_text" else None
        )
        self._clip_text_buffer_names: dict[str, str] = {}
        self._clip_embeddings_ready: dict[str, bool] = {}
        if self.prompt_source == "clip_text":
            for dataset_id, num_classes in self.num_classes_by_id.items():
                buffer_name = f"clip_text_embeddings_{self._id_to_module_key[dataset_id]}"
                self.register_buffer(
                    buffer_name,
                    torch.zeros(num_classes, self.clip_embedding_dim, dtype=torch.float32),
                    persistent=True,
                )
                self._clip_text_buffer_names[dataset_id] = buffer_name
                self._clip_embeddings_ready[dataset_id] = False

    @staticmethod
    def _safe_module_key(dataset_id: str) -> str:
        return re.sub(r"[^0-9A-Za-z_]", "_", str(dataset_id))

    @staticmethod
    def _normalize_prompt_source(prompt_source: str) -> str:
        prompt_source = str(prompt_source or "learned").strip().lower()
        aliases = {
            "learned_embedding": "learned",
            "learned_embeddings": "learned",
            "rand_embedding": "learned",
            "random_embedding": "learned",
            "clip": "clip_text",
            "clip_encoder": "clip_text",
            "word_embedding": "clip_text",
            "word_embeddings": "clip_text",
        }
        prompt_source = aliases.get(prompt_source, prompt_source)
        if prompt_source not in {"learned", "clip_text"}:
            raise ValueError(
                "MT_CLIP_PROMPT_SOURCE must be 'learned' or 'clip_text' "
                f"(got {prompt_source!r})."
            )
        return prompt_source

    @staticmethod
    def _label_sort_value(value: Any) -> int:
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, (list, tuple)):
            numeric = [int(i) for i in value if isinstance(i, (int, float))]
            return min(numeric) if numeric else 0
        return 0

    def update_prompt_texts_from_dataset_jsons(self, dataset_jsons: dict[str, dict]) -> None:
        prompt_text_overrides = self._load_prompt_text_overrides()
        for dataset_id, dataset_json in dataset_jsons.items():
            dataset_id = str(dataset_id)
            if dataset_id not in self.num_classes_by_id:
                continue
            labels = dataset_json.get("labels", {})
            ordered_labels = sorted(labels.items(), key=lambda item: self._label_sort_value(item[1]))
            prompt_texts = [
                self._format_prompt_text(label_name, dataset_id, label_index)
                for label_index, (label_name, _) in enumerate(ordered_labels)
            ]
            if dataset_id in prompt_text_overrides:
                prompt_texts = prompt_text_overrides[dataset_id]
            expected = self.num_classes_by_id[dataset_id]
            if len(prompt_texts) < expected:
                prompt_texts.extend(
                    [self._format_prompt_text(f"class_{i}", dataset_id, i) for i in range(len(prompt_texts), expected)]
                )
            prompt_texts = [str(text) for text in prompt_texts[:expected]]
            self.prompt_texts_by_id[dataset_id] = prompt_texts
            if self.prompt_source == "clip_text":
                self._refresh_clip_text_embeddings(dataset_id, prompt_texts)

    def _format_prompt_text(self, label_name: str, dataset_id: str, label_index: int) -> str:
        clean_label = str(label_name).replace("_", " ")
        try:
            return self.clip_prompt_template.format(
                label=clean_label,
                dataset_id=str(dataset_id),
                index=int(label_index),
            )
        except KeyError as exc:
            raise KeyError(
                "MT_CLIP_PROMPT_TEMPLATE supports only {label}, {dataset_id}, and {index} placeholders."
            ) from exc

    def _load_prompt_text_overrides(self) -> dict[str, list[str]]:
        if not self.clip_prompt_texts_json:
            return {}
        prompt_json_path = Path(self.clip_prompt_texts_json).expanduser()
        with prompt_json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "prompt_texts_by_id" in payload:
            payload = payload["prompt_texts_by_id"]
        if not isinstance(payload, dict):
            raise ValueError(
                "MT_CLIP_PROMPT_TEXTS_JSON must contain a mapping from dataset id to prompt text list."
            )
        return {str(dataset_id): [str(text) for text in texts] for dataset_id, texts in payload.items()}

    def _refresh_clip_text_embeddings(self, dataset_id: str, prompt_texts: list[str]) -> None:
        cached_embeddings = self._load_cached_clip_embeddings(dataset_id, prompt_texts)
        embeddings = cached_embeddings if cached_embeddings is not None else self._encode_clip_prompt_texts(prompt_texts)
        buffer_name = self._clip_text_buffer_names[dataset_id]
        buffer = getattr(self, buffer_name)
        if tuple(embeddings.shape) != tuple(buffer.shape):
            raise ValueError(
                f"CLIP prompt embeddings for dataset {dataset_id} have shape {tuple(embeddings.shape)}, "
                f"expected {tuple(buffer.shape)}."
            )
        with torch.no_grad():
            buffer.copy_(embeddings.to(device=buffer.device, dtype=buffer.dtype))
        self._clip_embeddings_ready[dataset_id] = True
        if cached_embeddings is None:
            self._save_cached_clip_embeddings(dataset_id, prompt_texts, embeddings)

    def _encode_clip_prompt_texts(self, prompt_texts: list[str]) -> torch.Tensor:
        try:
            import clip
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MT_CLIP_PROMPT_SOURCE=clip_text requires OpenAI CLIP. Install it with "
                "`python -m pip install git+https://github.com/openai/CLIP.git` or provide a cached "
                "embedding file through MT_CLIP_PROMPT_CACHE."
            ) from exc

        device = self.clip_device or "cpu"
        model, _ = clip.load(self.clip_model_name, device=device)
        model.eval()
        tokens = clip.tokenize(prompt_texts).to(device)
        with torch.no_grad():
            embeddings = model.encode_text(tokens).float()
            embeddings = F.normalize(embeddings, dim=-1)
        return embeddings.cpu()

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
        entry = entries.get(str(dataset_id))
        if entry is None:
            return None
        if isinstance(entry, dict):
            if entry.get("clip_model_name", self.clip_model_name) != self.clip_model_name:
                return None
            if entry.get("prompt_texts") != prompt_texts:
                return None
            entry = entry.get("embeddings")
        if entry is None:
            return None
        embeddings = torch.as_tensor(entry, dtype=torch.float32)
        return embeddings

    def _save_cached_clip_embeddings(self, dataset_id: str, prompt_texts: list[str], embeddings: torch.Tensor) -> None:
        if not self.clip_prompt_cache:
            return
        cache_path = Path(self.clip_prompt_cache).expanduser()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"embeddings_by_id": {}}
        if cache_path.is_file():
            try:
                existing = self._torch_load(cache_path)
                if isinstance(existing, dict):
                    payload = existing
            except Exception:
                payload = {"embeddings_by_id": {}}
        if "embeddings_by_id" not in payload or not isinstance(payload["embeddings_by_id"], dict):
            payload["embeddings_by_id"] = {}
        payload["embeddings_by_id"][str(dataset_id)] = {
            "clip_model_name": self.clip_model_name,
            "prompt_texts": list(prompt_texts),
            "embeddings": embeddings.detach().cpu(),
        }
        torch.save(payload, cache_path)

    def forward(self, outputs, ids: list[str]):
        if not self.enable_clip_prompt and not self.enable_dynamic_conv:
            return outputs
        return self._apply_outputs(outputs, ids)

    def _apply_outputs(self, outputs, ids: list[str]):
        if isinstance(outputs, list):
            if len(outputs) == len(ids):
                return [self._apply_one_sample_output(o, ids[i]) for i, o in enumerate(outputs)]
            if len(ids) == 1:
                return self._apply_one_sample_output(outputs, ids[0])
            return [self._apply_one_sample_output(o, ids[min(i, len(ids) - 1)]) for i, o in enumerate(outputs)]
        if isinstance(outputs, tuple):
            return tuple(self._apply_outputs(list(outputs), ids))
        if torch.is_tensor(outputs):
            return self._adapt_logits(outputs, ids[0])
        return outputs

    def _apply_one_sample_output(self, output, dataset_id: str):
        if torch.is_tensor(output):
            return self._adapt_logits(output, dataset_id)
        if isinstance(output, list):
            return [self._apply_one_sample_output(o, dataset_id) for o in output]
        if isinstance(output, tuple):
            return tuple(self._apply_one_sample_output(o, dataset_id) for o in output)
        return output

    def _adapt_logits(self, logits: torch.Tensor, dataset_id: str) -> torch.Tensor:
        if logits.ndim not in (4, 5):
            return logits

        dataset_id = str(dataset_id)
        if dataset_id not in self.num_classes_by_id:
            raise KeyError(f"No prompt/dynamic head was configured for dataset id {dataset_id}")

        num_logits_channels = logits.shape[1]
        expected_channels = self.num_classes_by_id[dataset_id]
        if num_logits_channels > expected_channels:
            raise ValueError(
                f"Logits for {dataset_id} have {num_logits_channels} channels, "
                f"but only {expected_channels} prompts are configured."
            )

        conditioning = self._build_conditioning(logits, dataset_id, num_logits_channels)
        adapted = logits

        if self.enable_clip_prompt:
            film = self.prompt_film(conditioning.reshape(-1, self.prompt_dim)).reshape(
                logits.shape[0], num_logits_channels, 2
            )
            view_shape = (logits.shape[0], num_logits_channels, *([1] * (logits.ndim - 2)))
            scale = torch.tanh(film[..., 0]).reshape(view_shape)
            bias = film[..., 1].reshape(view_shape)
            gate = torch.sigmoid(self.prompt_gate_logit).to(device=logits.device, dtype=logits.dtype)
            adapted = adapted + gate * (scale * adapted + bias)

        if self.enable_dynamic_conv:
            params = self.controller(conditioning.reshape(-1, self.prompt_dim))
            params = torch.tanh(params) * self.controller_param_scale
            dynamic_logits = self._run_dynamic_head(adapted, params)
            dynamic_logits = torch.tanh(dynamic_logits) * self.dynamic_logit_scale
            gate = torch.sigmoid(self.dynamic_gate_logit).to(device=logits.device, dtype=logits.dtype)
            adapted = adapted + gate * dynamic_logits

        return adapted

    def _build_conditioning(self, logits: torch.Tensor, dataset_id: str, num_logits_channels: int) -> torch.Tensor:
        class_prompts = self._get_class_prompts(dataset_id, num_logits_channels)
        class_prompts = class_prompts.to(device=logits.device, dtype=logits.dtype)

        batch_size = logits.shape[0]
        spatial_dims = tuple(range(2, logits.ndim))
        image_context = logits.mean(dim=(1, *spatial_dims), keepdim=False).reshape(batch_size, 1)
        image_context = self.context_projector(image_context).to(dtype=logits.dtype)
        return class_prompts.unsqueeze(0) + image_context.unsqueeze(1)

    def _get_class_prompts(self, dataset_id: str, num_logits_channels: int) -> torch.Tensor:
        module_key = self._id_to_module_key[dataset_id]
        if self.prompt_source == "clip_text":
            buffer = getattr(self, self._clip_text_buffer_names[dataset_id])
            if not self._clip_embeddings_ready.get(dataset_id, False) and torch.count_nonzero(buffer.detach()).item():
                self._clip_embeddings_ready[dataset_id] = True
            if not self._clip_embeddings_ready.get(dataset_id, False):
                raise RuntimeError(
                    f"CLIP prompt embeddings for dataset {dataset_id} are not initialized. "
                    "Call update_prompt_texts_from_dataset_jsons before training or prediction."
                )
            if self.clip_text_to_prompt is None:
                raise RuntimeError("clip_text_to_prompt is missing although prompt_source is clip_text.")
            return self.clip_text_to_prompt(buffer[:num_logits_channels])
        return self.class_embeddings[module_key].weight[:num_logits_channels]

    def _run_dynamic_head(self, logits: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels = logits.shape[:2]
        spatial_shape = logits.shape[2:]
        num_instances = batch_size * num_channels
        grouped_features = logits.reshape(1, num_instances, *spatial_shape)

        weights, biases = self._parse_dynamic_params(params, logits.ndim)
        conv_op = F.conv3d if logits.ndim == 5 else F.conv2d
        x = grouped_features
        for layer_idx, (weight, bias) in enumerate(zip(weights, biases)):
            x = conv_op(x, weight, bias=bias, stride=1, padding=0, groups=num_instances)
            if layer_idx < len(weights) - 1:
                x = F.relu(x, inplace=True)
        return x.reshape(batch_size, num_channels, *spatial_shape)

    def _parse_dynamic_params(self, params: torch.Tensor, logit_ndim: int):
        if params.dim() != 2:
            raise ValueError(f"Expected dynamic params to be 2D, got shape {tuple(params.shape)}")
        if params.size(1) != self.num_dynamic_params:
            raise ValueError(
                f"Expected {self.num_dynamic_params} dynamic params, got {params.size(1)}"
            )

        num_instances = params.size(0)
        params_splits = torch.split(params, self.weight_nums + self.bias_nums, dim=1)
        weight_splits = list(params_splits[:len(self.weight_nums)])
        bias_splits = list(params_splits[len(self.weight_nums):])
        kernel = (1,) * (logit_ndim - 2)

        weight_splits[0] = weight_splits[0].reshape(num_instances * self.dynamic_hidden_channels, 1, *kernel)
        bias_splits[0] = bias_splits[0].reshape(num_instances * self.dynamic_hidden_channels)
        weight_splits[1] = weight_splits[1].reshape(
            num_instances * self.dynamic_hidden_channels, self.dynamic_hidden_channels, *kernel
        )
        bias_splits[1] = bias_splits[1].reshape(num_instances * self.dynamic_hidden_channels)
        weight_splits[2] = weight_splits[2].reshape(num_instances, self.dynamic_hidden_channels, *kernel)
        bias_splits[2] = bias_splits[2].reshape(num_instances)

        return weight_splits, bias_splits

    def get_metadata(self) -> dict[str, Any]:
        metadata = {
            "num_classes_by_id": self.num_classes_by_id,
            "enable_clip_prompt": self.enable_clip_prompt,
            "enable_dynamic_conv": self.enable_dynamic_conv,
            "prompt_dim": self.prompt_dim,
            "prompt_source": self.prompt_source,
            "clip_model_name": self.clip_model_name if self.prompt_source == "clip_text" else None,
            "clip_prompt_template": self.clip_prompt_template if self.prompt_source == "clip_text" else None,
            "clip_prompt_texts_json": self.clip_prompt_texts_json or None,
            "clip_prompt_cache": self.clip_prompt_cache or None,
            "clip_embeddings_ready": dict(self._clip_embeddings_ready),
            "dynamic_hidden_channels": self.dynamic_hidden_channels,
            "controller_param_scale": self.controller_param_scale,
            "dynamic_logit_scale": self.dynamic_logit_scale,
            "prompt_texts_by_id": self.prompt_texts_by_id,
            "reference": "CLIP-Driven-Universal-Model organ_embedding/controller dynamic heads",
        }
        if self.enable_clip_prompt:
            metadata["prompt_gate"] = float(torch.sigmoid(self.prompt_gate_logit).detach().cpu())
        if self.enable_dynamic_conv:
            metadata["dynamic_gate"] = float(torch.sigmoid(self.dynamic_gate_logit).detach().cpu())
        return metadata


class MultiTalentAblationWrapper(nn.Module):
    """Composable ablation wrapper around a MultiTalent-compatible network."""

    def __init__(
            self,
            base_network: nn.Module,
            num_classes_by_id: dict[str, int],
            enable_clip_prompt: bool = False,
            enable_uniseg_prompt: bool = False,
            enable_dynamic_conv: bool = False,
    ):
        super().__init__()
        self.base_network = base_network
        self.num_classes_by_id = {str(k): int(v) for k, v in num_classes_by_id.items()}
        self.enable_clip_prompt = bool(enable_clip_prompt)
        self.enable_uniseg_prompt = bool(enable_uniseg_prompt)
        self.enable_dynamic_conv = bool(enable_dynamic_conv)

        self.logit_adapter = None
        if self.enable_clip_prompt or self.enable_dynamic_conv:
            self.logit_adapter = ClipDynamicLogitAdapter(
                self.num_classes_by_id,
                enable_clip_prompt=self.enable_clip_prompt,
                enable_dynamic_conv=self.enable_dynamic_conv,
            )

        self.uniseg_adapter = None
        if self.enable_uniseg_prompt:
            conv_op = getattr(self.base_network.encoder, "conv_op", None)
            output_channels = getattr(self.base_network.encoder, "output_channels", None)
            if conv_op is None or output_channels is None:
                raise ValueError("UniSeg prompt requires a base network with encoder.conv_op and encoder.output_channels")
            self.uniseg_adapter = UniSegBottleneckPromptAdapter(
                list(self.num_classes_by_id.keys()),
                int(output_channels[-1]),
                conv_op,
            )

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
        return [str(i) for i in ids]

    def update_prompt_texts_from_dataset_jsons(self, dataset_jsons: dict[str, dict]) -> None:
        if self.logit_adapter is not None:
            self.logit_adapter.update_prompt_texts_from_dataset_jsons(dataset_jsons)

    def forward(self, x, ids):
        ids = self._normalize_ids(ids)
        if self.uniseg_adapter is not None:
            skips = list(self.base_network.encoder(x, ids))
            skips[-1] = self.uniseg_adapter(skips[-1], ids)
            outputs = self.base_network.decoder(skips, ids)
        else:
            outputs = self.base_network(x, ids)

        if self.logit_adapter is not None:
            outputs = self.logit_adapter(outputs, ids)
        return outputs

    def compute_conv_feature_map_size(self, input_size):
        return self.base_network.compute_conv_feature_map_size(input_size)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        has_wrapper_keys = any(k.startswith("base_network.") for k in state_dict.keys())
        if has_wrapper_keys:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)

        own_keys = self.state_dict().keys()
        remapped = {}
        for key, value in state_dict.items():
            base_key = f"base_network.{key}"
            remapped[base_key if base_key in own_keys else key] = value
        return super().load_state_dict(remapped, strict=False, assign=assign)

    def get_ablation_metadata(self) -> dict[str, Any]:
        metadata = {
            "base_network": self.base_network.__class__.__name__,
            "num_classes_by_id": self.num_classes_by_id,
            "enable_clip_prompt": self.enable_clip_prompt,
            "enable_uniseg_prompt": self.enable_uniseg_prompt,
            "enable_dynamic_conv": self.enable_dynamic_conv,
        }
        if self.logit_adapter is not None:
            metadata["clip_dynamic_logit_adapter"] = self.logit_adapter.get_metadata()
        if self.uniseg_adapter is not None:
            metadata["uniseg_bottleneck_prompt_adapter"] = self.uniseg_adapter.get_metadata()
        return metadata
