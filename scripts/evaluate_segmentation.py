#!/usr/bin/env python3
"""Compute overlap and surface metrics for aligned multiclass NIfTI segmentations."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Folder with ground-truth NIfTI files")
    parser.add_argument("--prediction", type=Path, required=True, help="Folder with matching prediction files")
    parser.add_argument("--labels", type=int, nargs="+", required=True, help="Foreground label values")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON report")
    parser.add_argument("--file-ending", default=".nii.gz")
    return parser.parse_args()


def geometry(image: sitk.Image) -> dict:
    return {
        "size": tuple(image.GetSize()),
        "spacing": tuple(float(v) for v in image.GetSpacing()),
        "origin": tuple(float(v) for v in image.GetOrigin()),
        "direction": tuple(float(v) for v in image.GetDirection()),
    }


def check_geometry(reference: sitk.Image, prediction: sitk.Image, name: str) -> None:
    ref = geometry(reference)
    pred = geometry(prediction)
    if ref["size"] != pred["size"]:
        raise ValueError(f"Geometry mismatch for {name}: size {ref['size']} != {pred['size']}")
    for key in ("spacing", "origin", "direction"):
        if not np.allclose(ref[key], pred[key], rtol=0, atol=1e-5):
            raise ValueError(f"Geometry mismatch for {name}: {key} {ref[key]} != {pred[key]}")


def surface_distances(a: np.ndarray, b: np.ndarray, spacing_zyx: tuple[float, ...]) -> np.ndarray:
    structure = generate_binary_structure(a.ndim, 1)
    surface_a = a & ~binary_erosion(a, structure=structure, border_value=0)
    surface_b = b & ~binary_erosion(b, structure=structure, border_value=0)
    distance_to_b = distance_transform_edt(~surface_b, sampling=spacing_zyx)
    distance_to_a = distance_transform_edt(~surface_a, sampling=spacing_zyx)
    return np.concatenate([distance_to_b[surface_a], distance_to_a[surface_b]])


def metrics_for_masks(reference: np.ndarray, prediction: np.ndarray, spacing_zyx) -> dict:
    reference = reference.astype(bool, copy=False)
    prediction = prediction.astype(bool, copy=False)
    ref_count = int(reference.sum())
    pred_count = int(prediction.sum())
    intersection = int(np.logical_and(reference, prediction).sum())

    if ref_count == 0 and pred_count == 0:
        return {
            "status": "both_empty",
            "dice": 1.0,
            "jaccard": 1.0,
            "assd": 0.0,
            "hd95": 0.0,
            "precision": 1.0,
            "recall": 1.0,
            "reference_voxels": 0,
            "prediction_voxels": 0,
        }
    if ref_count == 0 or pred_count == 0:
        return {
            "status": "one_empty",
            "dice": 0.0,
            "jaccard": 0.0,
            "assd": None,
            "hd95": None,
            "precision": 0.0,
            "recall": 0.0,
            "reference_voxels": ref_count,
            "prediction_voxels": pred_count,
        }

    false_positive = pred_count - intersection
    false_negative = ref_count - intersection
    distances = surface_distances(reference, prediction, spacing_zyx)
    return {
        "status": "nonempty",
        "dice": 2.0 * intersection / (2.0 * intersection + false_positive + false_negative),
        "jaccard": intersection / (intersection + false_positive + false_negative),
        "assd": float(distances.mean()),
        "hd95": float(np.percentile(distances, 95)),
        "precision": intersection / (intersection + false_positive),
        "recall": intersection / (intersection + false_negative),
        "reference_voxels": ref_count,
        "prediction_voxels": pred_count,
    }


def finite_mean(values) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def aggregate(records: list[dict]) -> dict:
    metric_names = ("dice", "jaccard", "assd", "hd95", "precision", "recall")
    return {
        "count": len(records),
        "one_empty_count": sum(record["status"] == "one_empty" for record in records),
        **{name: finite_mean(record[name] for record in records) for name in metric_names},
    }


def collect_files(folder: Path, ending: str) -> dict[str, Path]:
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    files = {path.name: path for path in folder.iterdir() if path.is_file() and path.name.endswith(ending)}
    if not files:
        raise FileNotFoundError(f"No *{ending} files found in {folder}")
    return files


def main() -> None:
    args = parse_args()
    if len(set(args.labels)) != len(args.labels) or any(label <= 0 for label in args.labels):
        raise ValueError("--labels must contain unique positive foreground values.")

    reference_files = collect_files(args.reference, args.file_ending)
    prediction_files = collect_files(args.prediction, args.file_ending)
    missing = sorted(set(reference_files) - set(prediction_files))
    extra = sorted(set(prediction_files) - set(reference_files))
    if missing or extra:
        raise ValueError(f"Filename mismatch: missing_predictions={missing}, extra_predictions={extra}")

    records = []
    for name in sorted(reference_files):
        reference_image = sitk.ReadImage(str(reference_files[name]))
        prediction_image = sitk.ReadImage(str(prediction_files[name]))
        check_geometry(reference_image, prediction_image, name)
        reference = sitk.GetArrayFromImage(reference_image)
        prediction = sitk.GetArrayFromImage(prediction_image)
        spacing_zyx = tuple(reversed(reference_image.GetSpacing()))
        for label in args.labels:
            record = metrics_for_masks(reference == label, prediction == label, spacing_zyx)
            record.update({"case": name, "label": label})
            records.append(record)

    by_label = {
        str(label): aggregate([record for record in records if record["label"] == label])
        for label in args.labels
    }
    report = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "reference": str(args.reference),
        "prediction": str(args.prediction),
        "file_ending": args.file_ending,
        "labels": args.labels,
        "empty_mask_policy": {
            "both_empty": "overlap/precision/recall=1; ASSD/HD95=0",
            "one_empty": "overlap/precision/recall=0; ASSD/HD95=null and excluded from finite means",
        },
        "distance_units": "physical units from reference image spacing",
        "per_case_label": records,
        "macro_all_case_labels": aggregate(records),
        "macro_by_label": by_label,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"output={args.output}")
    print(json.dumps(report["macro_all_case_labels"], sort_keys=True))


if __name__ == "__main__":
    main()
