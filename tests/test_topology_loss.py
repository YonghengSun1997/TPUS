from __future__ import annotations

import unittest

import torch

from multitalent.training.loss.nextou_topology_losses import TI_Loss, make_pairwise_ti_exclusions


class TopologyLossTests(unittest.TestCase):
    def test_single_foreground_class_has_no_pairwise_constraint(self):
        exclusions = make_pairwise_ti_exclusions(2)
        self.assertEqual(exclusions, [])
        logits = torch.randn(1, 2, 4, 4, 4, requires_grad=True)
        target = torch.zeros(1, 1, 4, 4, 4, dtype=torch.long)
        loss = TI_Loss(dim=3, connectivity=26, exclusion=exclusions)(logits, target)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertEqual(logits.grad.abs().sum().item(), 0.0)

    def test_adjacent_excluded_labels_produce_finite_loss_and_gradient(self):
        exclusions = make_pairwise_ti_exclusions(5)
        self.assertEqual(len(exclusions), 6)
        logits = torch.full((1, 5, 4, 4, 4), -4.0)
        logits[:, 1, :, :, :2] = 4.0
        logits[:, 2, :, :, 2:] = 4.0
        logits.requires_grad_()
        target = torch.zeros(1, 1, 4, 4, 4, dtype=torch.long)
        loss = TI_Loss(dim=3, connectivity=26, exclusion=exclusions)(logits, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
