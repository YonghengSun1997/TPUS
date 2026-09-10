#!/usr/bin/env python3
"""Generate the raw OpenAI CLIP text cache consumed by the TPUS head."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode dataset-specific class prompts with raw OpenAI CLIP text features. "
            "Embeddings are deliberately not L2-normalized."
        )
    )
    parser.add_argument("--prompts", type=Path, required=True, help="JSON mapping dataset IDs to prompt lists")
    parser.add_argument("--output", type=Path, required=True, help="Output .pt cache")
    parser.add_argument("--model", default="ViT-B/32", help="OpenAI CLIP model name")
    parser.add_argument("--device", default="cpu", help="PyTorch device used only for text encoding")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output cache")
    return parser.parse_args()


def load_prompts(path: Path) -> dict[str, list[str]]:
    with path.expanduser().open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "prompt_texts_by_id" in payload:
        payload = payload["prompt_texts_by_id"]
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Prompt JSON must be a non-empty mapping from dataset IDs to text lists.")

    prompts: dict[str, list[str]] = {}
    for dataset_id, values in payload.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Dataset {dataset_id!r} must have a non-empty prompt list.")
        texts = [str(value).strip() for value in values]
        if any(not text for text in texts):
            raise ValueError(f"Dataset {dataset_id!r} contains an empty prompt.")
        prompts[str(dataset_id)] = texts
    return prompts


def main() -> None:
    args = parse_args()
    output = args.output.expanduser()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing cache: {output}")

    try:
        import clip
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install the optional CLIP dependency with: python -m pip install -e '.[clip]'") from exc

    prompts_by_id = load_prompts(args.prompts)
    model, _ = clip.load(args.model, device=args.device)
    model.eval()

    entries = {}
    with torch.no_grad():
        for dataset_id, prompt_texts in prompts_by_id.items():
            tokens = clip.tokenize(prompt_texts).to(args.device)
            embeddings = model.encode_text(tokens).float().cpu()
            entries[dataset_id] = {
                "clip_model_name": args.model,
                "prompt_texts": prompt_texts,
                "normalized": False,
                "embeddings": embeddings,
            }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.{os.getpid()}.tmp")
    torch.save({"embeddings_by_id": entries}, temporary)
    os.replace(temporary, output)

    print(f"saved={output}")
    for dataset_id, entry in entries.items():
        embeddings = entry["embeddings"]
        print(
            f"dataset={dataset_id} shape={tuple(embeddings.shape)} "
            f"norms={embeddings.norm(dim=-1).tolist()} normalized=false"
        )


if __name__ == "__main__":
    main()
