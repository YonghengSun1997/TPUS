#!/usr/bin/env python3
"""Generate small non-medical nnU-Net datasets for TPUS smoke tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk


DATASET_SPECS = {
    "Dataset901_SyntheticUS": {
        "channel_names": {"0": "synthetic_US"},
        "labels": {"background": 0, "synthetic_uterus": 1},
    },
    "Dataset902_SyntheticMRI": {
        "channel_names": {"0": "synthetic_T2_MRI"},
        "labels": {
            "background": 0,
            "synthetic_wall": 1,
            "synthetic_cavity": 2,
            "synthetic_lesion": 3,
            "synthetic_cyst": 4,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True, help="Parent directory for Dataset901/902")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--force", action="store_true", help="Replace existing synthetic Dataset901/902 folders")
    return parser.parse_args()


def ellipsoid(shape: tuple[int, int, int], center, radii) -> np.ndarray:
    coordinates = np.ogrid[tuple(slice(0, size) for size in shape)]
    distance = sum(((axis - c) / r) ** 2 for axis, c, r in zip(coordinates, center, radii))
    return distance <= 1.0


def synthetic_us(case_index: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    shape = (24, 32, 32)
    z, y, x = np.indices(shape)
    center = (12 + case_index % 2, 17, 16)
    radial = np.sqrt((x - 16) ** 2 + (y - 4) ** 2)
    fan = (y >= 4) & (radial <= 28) & (np.abs(x - 16) <= (y - 2) * 0.65)
    foreground = ellipsoid(shape, center, (6, 8, 10)) & fan
    image = 0.12 * rng.normal(size=shape) + 0.4 * np.exp(-radial / 20)
    image += foreground * 0.8
    image *= fan
    return image.astype(np.float32), foreground.astype(np.uint8)


def synthetic_mri(case_index: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    shape = (24, 32, 32)
    center = (12, 16, 16 + case_index % 2)
    wall_outer = ellipsoid(shape, center, (8, 11, 10))
    cavity = ellipsoid(shape, center, (4, 6, 5))
    lesion = ellipsoid(shape, (13, 19, 20), (3, 4, 3))
    cyst = ellipsoid(shape, (10, 13, 11), (2, 2, 2))

    segmentation = np.zeros(shape, dtype=np.uint8)
    segmentation[wall_outer] = 1
    segmentation[cavity] = 2
    segmentation[lesion] = 3
    segmentation[cyst] = 4

    image = 0.08 * rng.normal(size=shape)
    intensities = np.asarray([0.0, 0.55, 1.0, 0.25, 1.35], dtype=np.float32)
    image += intensities[segmentation]
    return image.astype(np.float32), segmentation


def write_volume(array: np.ndarray, path: Path, spacing: tuple[float, float, float]) -> None:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(image, str(path), useCompression=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_directory(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Refusing to overwrite existing synthetic dataset: {path}")
        shutil.rmtree(path)
    for name in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (path / name).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    manifest = {"synthetic": True, "seed": args.seed, "datasets": {}}

    generators = {
        "Dataset901_SyntheticUS": (synthetic_us, (0.8, 0.8, 1.5)),
        "Dataset902_SyntheticMRI": (synthetic_mri, (0.9, 0.9, 1.2)),
    }
    for dataset_name, spec in DATASET_SPECS.items():
        dataset_root = root / dataset_name
        prepare_directory(dataset_root, args.force)
        generator, spacing = generators[dataset_name]
        files = []
        for split, count in (("Tr", 2), ("Ts", 1)):
            for index in range(count):
                case_id = f"synthetic_{dataset_name[7:10]}_{split.lower()}_{index:03d}"
                image, label = generator(index, rng)
                image_path = dataset_root / f"images{split}" / f"{case_id}_0000.nii.gz"
                label_path = dataset_root / f"labels{split}" / f"{case_id}.nii.gz"
                write_volume(image, image_path, spacing)
                write_volume(label, label_path, spacing)
                files.extend([image_path, label_path])

        dataset_json = {
            **spec,
            "numTraining": 2,
            "file_ending": ".nii.gz",
            "overwrite_image_reader_writer": "SimpleITKIO",
            "synthetic_data": True,
        }
        dataset_json_path = dataset_root / "dataset.json"
        dataset_json_path.write_text(json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8")
        files.append(dataset_json_path)
        manifest["datasets"][dataset_name] = {
            "path": str(dataset_root),
            "files": {str(path.relative_to(root)): sha256(path) for path in sorted(files)},
        }

    manifest_path = root / "synthetic_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"root={root}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
