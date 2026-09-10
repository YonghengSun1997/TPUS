from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_segmentation import metrics_for_masks


class ExtendedMetricTests(unittest.TestCase):
    def test_identical_nonempty_masks(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[1:4, 1:4, 1:4] = True
        result = metrics_for_masks(mask, mask.copy(), (1.0, 1.0, 1.0))
        for name in ("dice", "jaccard", "precision", "recall"):
            self.assertEqual(result[name], 1.0)
        self.assertEqual(result["assd"], 0.0)
        self.assertEqual(result["hd95"], 0.0)

    def test_both_empty_and_one_empty_policy(self):
        empty = np.zeros((4, 4, 4), dtype=bool)
        object_mask = empty.copy()
        object_mask[1:3, 1:3, 1:3] = True
        both = metrics_for_masks(empty, empty, (1.0, 1.0, 1.0))
        one = metrics_for_masks(object_mask, empty, (1.0, 1.0, 1.0))
        self.assertEqual(both["status"], "both_empty")
        self.assertEqual(both["dice"], 1.0)
        self.assertEqual(one["status"], "one_empty")
        self.assertEqual(one["dice"], 0.0)
        self.assertIsNone(one["assd"])
        self.assertIsNone(one["hd95"])

    def test_spacing_changes_surface_distance_units(self):
        reference = np.zeros((7, 7, 7), dtype=bool)
        prediction = np.zeros_like(reference)
        reference[2:5, 2:5, 2:5] = True
        prediction[3:6, 2:5, 2:5] = True
        unit = metrics_for_masks(reference, prediction, (1.0, 1.0, 1.0))
        doubled = metrics_for_masks(reference, prediction, (2.0, 1.0, 1.0))
        self.assertGreater(doubled["assd"], unit["assd"])
        self.assertGreaterEqual(doubled["hd95"], unit["hd95"])


if __name__ == "__main__":
    unittest.main()
