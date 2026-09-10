#!/usr/bin/env python3
"""Audit TPUS Task 850 architecture invariants without running a full patch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
TRAINER_NAME = "MultiTalent_trainer_clip_ti_gnn_clipdynfeature_multistems"


def parse_dataset_json(value: str) -> tuple[str, Path]:
    dataset_id, separator, path = value.partition("=")
    if not separator or not dataset_id or not path:
        raise argparse.ArgumentTypeError("Use DATASET_ID=/path/to/dataset.json")
    return str(dataset_id), Path(path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, required=True, help="Combined MultiTalent plans JSON")
    parser.add_argument(
        "--dataset-json",
        type=parse_dataset_json,
        action="append",
        required=True,
        metavar="ID=PATH",
        help="Source dataset JSON; pass once per dataset",
    )
    parser.add_argument("--prompt-texts", type=Path, required=True, help="Dataset prompt JSON")
    parser.add_argument("--prompt-cache", type=Path, required=True, help="Raw CLIP embedding cache")
    parser.add_argument("--output", type=Path, required=True, help="Audit report JSON")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--expected-pool-gnn", type=int, default=7)
    parser.add_argument("--expected-swin-gnn", type=int, default=7)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    dataset_paths = dict(args.dataset_json)
    if len(dataset_paths) != len(args.dataset_json):
        raise ValueError("Each --dataset-json must use a unique dataset ID.")
    if len(dataset_paths) < 2:
        raise ValueError("The Task 850 audit requires at least two source datasets.")

    for path in [args.plans, args.prompt_texts, args.prompt_cache, *dataset_paths.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)

    os.environ["MT_CLIP_PROMPT_MODEL"] = "ViT-B/32"
    os.environ["MT_CLIP_PROMPT_DEVICE"] = "cpu"
    os.environ["MT_CLIP_PROMPT_TEXTS_JSON"] = str(args.prompt_texts.resolve())
    os.environ["MT_FAITHFUL_CLIP_PROMPT_CACHE"] = str(args.prompt_cache.resolve())

    from multitalent.training.nnUNetTrainer.project_specific.MultiTalent_native.MultiTalent_clip_driven_feature import (
        MultiTalent_trainer_clip_ti_gnn_clipdynfeature_multistems,
    )
    from multitalent.utilities.find_class_by_name import recursive_find_python_class

    plans = load_json(args.plans)
    configuration = plans["configurations"][args.configuration]
    architecture = configuration["architecture"]
    arch_kwargs = deepcopy(architecture["arch_kwargs"])
    arch_kwargs["patch_size"] = list(configuration["patch_size"])
    dataset_jsons = {dataset_id: load_json(path) for dataset_id, path in dataset_paths.items()}
    input_channels = {
        dataset_id: len(dataset_json["channel_names"])
        for dataset_id, dataset_json in dataset_jsons.items()
    }
    output_channels = {
        dataset_id: len(dataset_json["labels"])
        for dataset_id, dataset_json in dataset_jsons.items()
    }

    network = MultiTalent_trainer_clip_ti_gnn_clipdynfeature_multistems.build_network_architecture(
        architecture["network_class_name"],
        arch_kwargs,
        architecture["_kw_requires_import"],
        input_channels,
        output_channels,
        enable_deep_supervision=True,
    )
    network.update_prompt_texts_from_dataset_jsons(dataset_jsons)
    network.eval()

    discovered_class = recursive_find_python_class(
        str(REPO_ROOT / "multitalent/training/nnUNetTrainer"),
        TRAINER_NAME,
        "multitalent.training.nnUNetTrainer",
    )
    stems = network.encoder.stem
    expected_ids = set(dataset_jsons)
    stem_parameter_ids = {
        dataset_id: {id(parameter) for parameter in stem.parameters()}
        for dataset_id, stem in stems.items()
    }
    stem_input_channels = {
        dataset_id: next(
            module.in_channels
            for module in stem.modules()
            if isinstance(module, (nn.Conv2d, nn.Conv3d))
        )
        for dataset_id, stem in stems.items()
    }
    shared_stem_parameters = any(
        stem_parameter_ids[left] & stem_parameter_ids[right]
        for index, left in enumerate(sorted(stem_parameter_ids))
        for right in sorted(stem_parameter_ids)[index + 1 :]
    )

    head = network.dynamic_head
    module_counts: dict[str, int] = {}
    for module in network.modules():
        name = type(module).__name__
        module_counts[name] = module_counts.get(name, 0) + 1
    clip_norms = {
        dataset_id: getattr(head, head._clip_text_buffer_names[dataset_id]).norm(dim=-1).tolist()
        for dataset_id in sorted(head.num_classes_by_id)
    }
    expected_auxiliary_heads = len(architecture["arch_kwargs"]["n_conv_per_stage_decoder"]) - 1

    checks = {
        "trainer_discoverable": discovered_class is MultiTalent_trainer_clip_ti_gnn_clipdynfeature_multistems,
        "dataset_specific_stems": isinstance(stems, nn.ModuleDict),
        "stem_ids_match": set(stems.keys()) == expected_ids,
        "stem_parameters_not_shared": not shared_stem_parameters,
        "input_channels_match": input_channels == stem_input_channels,
        "class_heads_match": head.num_classes_by_id == output_channels,
        "bottleneck_projection_matches": tuple(head.GAP[-1].weight.shape) == (
            head.conditioning_dim,
            head.bottleneck_channels,
            1,
            1,
            1,
        ),
        "decoder_projection_matches": tuple(head.precls_conv[-1].weight.shape) == (
            head.dynamic_channels,
            head.decoder_channels,
            1,
            1,
            1,
        ),
        "text_projection_matches": tuple(head.text_to_vision.weight.shape) == (
            head.conditioning_dim,
            head.clip_embedding_dim,
        ),
        "controller_shape_matches": tuple(head.controller.weight.shape) == (
            head.num_dynamic_params,
            head.conditioning_dim * 2,
            1,
            1,
            1,
        ),
        "dynamic_params_153": head.num_dynamic_params == 153,
        "raw_clip_embeddings": all(
            any(abs(norm - 1.0) > 1e-3 for norm in norms) for norms in clip_norms.values()
        ),
        "fixed_final_heads_removed": not network.decoder.has_final_segmentation_heads,
        "auxiliary_heads_match": all(
            len(network.decoder.seg_layers[dataset_id]) == expected_auxiliary_heads
            for dataset_id in expected_ids
        ),
        "pool_gnn_count": module_counts.get("PoolGNNBlocks", 0) == args.expected_pool_gnn,
        "swin_gnn_count": module_counts.get("SwinGNNBlocks", 0) == args.expected_swin_gnn,
        "no_logit_adapter": module_counts.get("ClipDynamicLogitAdapter", 0) == 0,
        "no_dynamic_or_prompt_gate": not hasattr(head, "dynamic_gate_logit")
        and not hasattr(head, "prompt_gate_logit"),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Architecture audit failed: {failed}")

    report = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "trainer": TRAINER_NAME,
        "configuration": args.configuration,
        "inputs": {
            "plans": {"path": str(args.plans), "sha256": sha256(args.plans)},
            "dataset_jsons": {
                dataset_id: {"path": str(path), "sha256": sha256(path)}
                for dataset_id, path in sorted(dataset_paths.items())
            },
            "prompt_texts": {"path": str(args.prompt_texts), "sha256": sha256(args.prompt_texts)},
            "prompt_cache": {"path": str(args.prompt_cache), "sha256": sha256(args.prompt_cache)},
        },
        "patch_size": configuration["patch_size"],
        "input_channels": input_channels,
        "output_channels": output_channels,
        "raw_clip_norms": clip_norms,
        "implementation_sha256": {
            "feature_wrapper": sha256(
                REPO_ROOT / "multitalent/utilities/MultiTalent/clip_driven_feature_wrapper.py"
            ),
            "decoder": sha256(
                REPO_ROOT / "multitalent/network_architecture/nextou/multitalent_nextou.py"
            ),
            "trainer": sha256(
                REPO_ROOT
                / "multitalent/training/nnUNetTrainer/project_specific/MultiTalent_native/"
                / "MultiTalent_clip_driven_feature.py"
            ),
        },
        "module_counts": module_counts,
        "parameter_count": sum(parameter.numel() for parameter in network.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in network.parameters() if parameter.requires_grad
        ),
        "output_contract": {
            "primary": "dynamic full-resolution logits from the D4 feature",
            "auxiliary": [f"static D{i}" for i in range(3, -1, -1)],
            "training_order": ["dynamic D4", "static D3", "static D2", "static D1", "static D0"],
        },
        "checks": checks,
        "all_checks_passed": True,
        "full_patch_forward_run": False,
        "full_patch_forward_note": (
            "Architecture-only instantiation. Compact functional forward/backward is covered by synthetic tests."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"audit={args.output}")
    print("all_checks_passed=true")
    print(f"parameter_count={report['parameter_count']}")


if __name__ == "__main__":
    main()
