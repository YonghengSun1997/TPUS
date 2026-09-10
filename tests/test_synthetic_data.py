from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from scripts import generate_synthetic_datasets as generator


class SyntheticDataTests(unittest.TestCase):
    def test_generators_are_aligned_and_use_declared_labels(self):
        rng = np.random.default_rng(123)
        for function, expected_labels in (
            (generator.synthetic_us, {0, 1}),
            (generator.synthetic_mri, {0, 1, 2, 3, 4}),
        ):
            image, label = function(0, rng)
            self.assertEqual(image.shape, label.shape)
            self.assertEqual(image.dtype, np.float32)
            self.assertEqual(label.dtype, np.uint8)
            self.assertEqual(set(np.unique(label).tolist()), expected_labels)
            self.assertTrue(np.isfinite(image).all())

    def test_nifti_roundtrip_preserves_geometry(self):
        with tempfile.TemporaryDirectory(prefix="tpus_nifti_") as temporary:
            path = Path(temporary) / "test.nii.gz"
            array = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
            generator.write_volume(array, path, (0.8, 0.9, 1.2))
            image = sitk.ReadImage(str(path))
            np.testing.assert_array_equal(sitk.GetArrayFromImage(image), array)
            self.assertTrue(np.allclose(image.GetSpacing(), (0.8, 0.9, 1.2)))

    def test_dataset_specs_are_explicitly_synthetic(self):
        for name, spec in generator.DATASET_SPECS.items():
            self.assertIn("Synthetic", name)
            self.assertEqual(spec["labels"]["background"], 0)
            self.assertIn("synthetic", json.dumps(spec).lower())


if __name__ == "__main__":
    unittest.main()
