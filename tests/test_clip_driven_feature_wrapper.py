from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from multitalent.network_architecture.nextou.multitalent_nextou import MultiTalentNexToUDecoder
from multitalent.utilities.MultiTalent.clip_driven_feature_wrapper import (
    ClipDrivenFeatureMultiTalentWrapper,
)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_op = nn.Conv3d
        self.output_channels = [32, 320]
        self.stem = nn.ModuleDict(
            {
                "901": nn.Conv3d(1, 32, kernel_size=3, padding=1),
                "902": nn.Conv3d(1, 32, kernel_size=3, padding=1),
            }
        )
        self.shared_bottleneck = nn.Conv3d(32, 320, kernel_size=1)

    def forward(self, x, ids):
        full_resolution = torch.cat(
            [self.stem[str(dataset_id)](sample) for sample, dataset_id in zip(x, ids)], dim=0
        )
        bottleneck = self.shared_bottleneck(F.avg_pool3d(full_resolution, kernel_size=4))
        return [full_resolution, bottleneck]


class TinyFeatureDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deep_supervision = True
        self.has_final_segmentation_heads = True
        self.final_static_heads = nn.ModuleDict(
            {"901": nn.Conv3d(32, 2, 1), "902": nn.Conv3d(32, 5, 1)}
        )
        self.auxiliary_heads = nn.ModuleDict(
            {"901": nn.Conv3d(32, 2, 1), "902": nn.Conv3d(32, 5, 1)}
        )

    def remove_final_segmentation_heads(self):
        self.final_static_heads = nn.ModuleDict()
        self.has_final_segmentation_heads = False

    def forward_features(self, skips, ids):
        final_feature = skips[0]
        auxiliary_outputs = []
        for index, dataset_id in enumerate(ids):
            feature = F.avg_pool3d(final_feature[index : index + 1], kernel_size=2)
            auxiliary_outputs.append([self.auxiliary_heads[str(dataset_id)](feature)])
        return final_feature, auxiliary_outputs


class TinyBaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyEncoder()
        self.decoder = TinyFeatureDecoder()

    @staticmethod
    def compute_conv_feature_map_size(input_size):
        return 0


def explicit_per_class_head(features: torch.Tensor, params: torch.Tensor, channels: int) -> torch.Tensor:
    """Evaluate each generated head independently, without grouped convolution."""
    outputs = []
    weight_sizes = (channels * channels, channels * channels, channels)
    bias_sizes = (channels, channels, 1)
    for class_index in range(params.shape[0]):
        vector = params[class_index]
        offset = 0
        weights = []
        for layer_index, size in enumerate(weight_sizes):
            chunk = vector[offset : offset + size]
            offset += size
            out_channels = channels if layer_index < 2 else 1
            weights.append(chunk.reshape(out_channels, channels, 1, 1, 1))
        biases = []
        for size in bias_sizes:
            biases.append(vector[offset : offset + size])
            offset += size
        if offset != vector.numel():
            raise AssertionError("The explicit head did not consume all generated parameters.")

        value = features[:, class_index * channels : (class_index + 1) * channels]
        for layer_index, (weight, bias) in enumerate(zip(weights, biases)):
            value = F.conv3d(value, weight, bias)
            if layer_index < 2:
                value = F.relu(value)
        outputs.append(value)
    return torch.cat(outputs, dim=1)


class ClipDrivenFeatureWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix="tpus_test_")
        cls.root = Path(cls.tempdir.name)
        cls.prompt_texts = {
            "901": ["synthetic background", "synthetic object"],
            "902": [
                "synthetic background",
                "synthetic region one",
                "synthetic region two",
                "synthetic region three",
                "synthetic region four",
            ],
        }
        cls.prompt_json = cls.root / "prompts.json"
        cls.prompt_json.write_text(json.dumps(cls.prompt_texts), encoding="utf-8")

        embeddings_by_id = {}
        for dataset_id, texts in cls.prompt_texts.items():
            values = torch.arange(1, len(texts) * 512 + 1, dtype=torch.float32).reshape(len(texts), 512)
            embeddings_by_id[dataset_id] = {
                "clip_model_name": "ViT-B/32",
                "prompt_texts": texts,
                "normalized": False,
                "embeddings": values / 100.0,
            }
        cls.prompt_cache = cls.root / "synthetic_prompt_cache.pt"
        torch.save({"embeddings_by_id": embeddings_by_id}, cls.prompt_cache)

        cls.original_environment = {
            name: os.environ.get(name)
            for name in (
                "MT_CLIP_PROMPT_TEXTS_JSON",
                "MT_FAITHFUL_CLIP_PROMPT_CACHE",
                "MT_CLIP_PROMPT_MODEL",
                "MT_CLIP_PROMPT_DEVICE",
            )
        }
        os.environ["MT_CLIP_PROMPT_TEXTS_JSON"] = str(cls.prompt_json)
        os.environ["MT_FAITHFUL_CLIP_PROMPT_CACHE"] = str(cls.prompt_cache)
        os.environ["MT_CLIP_PROMPT_MODEL"] = "ViT-B/32"
        os.environ["MT_CLIP_PROMPT_DEVICE"] = "cpu"

    @classmethod
    def tearDownClass(cls):
        for name, value in cls.original_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        cls.tempdir.cleanup()

    def _build_wrapper(self):
        wrapper = ClipDrivenFeatureMultiTalentWrapper(TinyBaseNetwork(), {"901": 2, "902": 5})
        wrapper.update_prompt_texts_from_dataset_jsons({"901": {}, "902": {}})
        return wrapper

    def test_architecture_and_raw_clip_cache(self):
        wrapper = self._build_wrapper()
        head = wrapper.dynamic_head

        self.assertEqual(head.num_dynamic_params, 153)
        self.assertEqual(tuple(head.controller.weight.shape), (153, 512, 1, 1, 1))
        self.assertEqual(tuple(head.GAP[-1].weight.shape), (256, 320, 1, 1, 1))
        self.assertEqual(tuple(head.precls_conv[-1].weight.shape), (8, 32, 1, 1, 1))
        self.assertFalse(wrapper.decoder.has_final_segmentation_heads)
        self.assertEqual(len(wrapper.decoder.final_static_heads), 0)
        self.assertFalse(hasattr(head, "dynamic_gate_logit"))
        self.assertFalse(hasattr(head, "prompt_gate_logit"))

        for dataset_id in ("901", "902"):
            buffer = getattr(head, head._clip_text_buffer_names[dataset_id])
            self.assertFalse(buffer.requires_grad)
            self.assertFalse(torch.allclose(buffer.norm(dim=-1), torch.ones(buffer.shape[0]), atol=1e-3))

    def test_multidataset_forward_and_backward(self):
        torch.manual_seed(7)
        wrapper = self._build_wrapper()
        inputs = [torch.randn(1, 1, 8, 8, 8), torch.randn(1, 1, 8, 8, 8)]
        outputs = wrapper(inputs, ["901", "902"])

        self.assertEqual(len(outputs), 2)
        self.assertEqual([tuple(t.shape) for t in outputs[0]], [(1, 2, 8, 8, 8), (1, 2, 4, 4, 4)])
        self.assertEqual([tuple(t.shape) for t in outputs[1]], [(1, 5, 8, 8, 8), (1, 5, 4, 4, 4)])

        loss = sum(t.square().mean() for sample_outputs in outputs for t in sample_outputs)
        loss.backward()
        parameters_requiring_grad = {
            "stem_901": wrapper.encoder.stem["901"].weight,
            "stem_902": wrapper.encoder.stem["902"].weight,
            "shared_bottleneck": wrapper.encoder.shared_bottleneck.weight,
            "gap_projection": wrapper.dynamic_head.GAP[-1].weight,
            "text_projection": wrapper.dynamic_head.text_to_vision.weight,
            "controller": wrapper.dynamic_head.controller.weight,
            "precls_projection": wrapper.dynamic_head.precls_conv[-1].weight,
        }
        for name, parameter in parameters_requiring_grad.items():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0, name)

        stem_overlap = {id(parameter) for parameter in wrapper.encoder.stem["901"].parameters()} & {
            id(parameter) for parameter in wrapper.encoder.stem["902"].parameters()
        }
        self.assertFalse(stem_overlap)

    def test_grouped_dynamic_head_matches_independent_per_class_evaluation(self):
        wrapper = self._build_wrapper()
        head = wrapper.dynamic_head
        torch.manual_seed(11)
        params = torch.randn(5, head.num_dynamic_params)
        features = torch.randn(1, 5 * head.dynamic_channels, 3, 3, 3)

        weights, biases = head.parse_dynamic_params(params)
        grouped = head.heads_forward(features.clone(), weights, biases, 5)
        explicit = explicit_per_class_head(features.clone(), params, head.dynamic_channels)
        torch.testing.assert_close(grouped, explicit)

    def test_checkpoint_roundtrip_restores_frozen_prompt_buffers(self):
        torch.manual_seed(19)
        source = self._build_wrapper().eval()
        inputs = [torch.randn(1, 1, 8, 8, 8), torch.randn(1, 1, 8, 8, 8)]
        with torch.no_grad():
            expected = source(inputs, ["901", "902"])

        restored = ClipDrivenFeatureMultiTalentWrapper(TinyBaseNetwork(), {"901": 2, "902": 5}).eval()
        restored.load_state_dict(source.state_dict(), strict=True)
        with torch.no_grad():
            actual = restored(inputs, ["901", "902"])

        for expected_sample, actual_sample in zip(expected, actual):
            for expected_tensor, actual_tensor in zip(expected_sample, actual_sample):
                torch.testing.assert_close(actual_tensor, expected_tensor)
        self.assertEqual(restored.dynamic_head._clip_embeddings_ready, {"901": True, "902": True})

    def test_train_and_eval_modes_propagate(self):
        wrapper = self._build_wrapper()
        wrapper.train()
        self.assertTrue(wrapper.training)
        self.assertTrue(wrapper.encoder.training)
        self.assertTrue(wrapper.dynamic_head.training)
        wrapper.eval()
        self.assertFalse(wrapper.training)
        self.assertFalse(wrapper.encoder.training)
        self.assertFalse(wrapper.dynamic_head.training)

    def test_decoder_feature_interface_removes_only_primary_static_head(self):
        decoder = MultiTalentNexToUDecoder.__new__(MultiTalentNexToUDecoder)
        nn.Module.__init__(decoder)
        decoder.deep_supervision = True
        decoder.transpconvs = nn.ModuleList(
            [nn.ConvTranspose3d(8, 4, 2, 2), nn.ConvTranspose3d(4, 2, 2, 2)]
        )
        decoder.stages = nn.ModuleList([nn.Conv3d(8, 4, 1), nn.Conv3d(4, 2, 1)])
        decoder.seg_layers = nn.ModuleDict(
            {
                "901": nn.ModuleList([nn.Conv3d(4, 2, 1), nn.Conv3d(2, 2, 1)]),
                "902": nn.ModuleList([nn.Conv3d(4, 5, 1), nn.Conv3d(2, 5, 1)]),
            }
        )
        decoder.has_final_segmentation_heads = True
        skips = [
            torch.randn(2, 2, 8, 8, 8),
            torch.randn(2, 4, 4, 4, 4),
            torch.randn(2, 8, 2, 2, 2),
        ]

        original_outputs = decoder(skips, ["901", "902"])
        self.assertEqual(tuple(original_outputs[0][0].shape), (1, 2, 8, 8, 8))
        self.assertEqual(tuple(original_outputs[1][0].shape), (1, 5, 8, 8, 8))

        decoder.remove_final_segmentation_heads()
        final_feature, auxiliary_outputs = decoder.forward_features(skips, ["901", "902"])
        self.assertEqual(tuple(final_feature.shape), (2, 2, 8, 8, 8))
        self.assertEqual(tuple(auxiliary_outputs[0][0].shape), (1, 2, 4, 4, 4))
        self.assertEqual(tuple(auxiliary_outputs[1][0].shape), (1, 5, 4, 4, 4))
        self.assertEqual(len(decoder.seg_layers["901"]), 1)
        self.assertEqual(len(decoder.seg_layers["902"]), 1)
        with self.assertRaisesRegex(RuntimeError, "forward_features"):
            decoder(skips, ["901", "902"])


if __name__ == "__main__":
    unittest.main()
