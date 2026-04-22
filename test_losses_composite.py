import unittest

import torch

from losses.components import ElevationL1Loss, SlopeL1Loss
from losses.composite import CompositeLoss


class CompositeLossTests(unittest.TestCase):
    def test_elevation_component_zero_when_equal(self) -> None:
        comp = ElevationL1Loss(weight=1.0)
        z_hat = torch.ones((2, 1, 4, 4))
        z_gt = torch.ones((2, 1, 4, 4))
        w = torch.ones((2, 1, 4, 4))
        value = comp(z_hat, z_gt, w)
        self.assertAlmostEqual(float(value), 0.0, places=7)

    def test_composite_loss_combines_components(self) -> None:
        loss_fn = CompositeLoss([ElevationL1Loss(weight=1.0), SlopeL1Loss(weight=0.5)])
        outputs = {"z_hat": torch.zeros((1, 1, 4, 4), dtype=torch.float32)}
        batch = {
            "z_gt": torch.ones((1, 1, 4, 4), dtype=torch.float32),
            "w": torch.ones((1, 1, 4, 4), dtype=torch.float32),
        }
        bundle = loss_fn(outputs, batch)
        self.assertIn("elev", bundle.metrics)
        self.assertIn("slope", bundle.metrics)
        self.assertGreater(float(bundle.loss), 0.0)


if __name__ == "__main__":
    unittest.main()

