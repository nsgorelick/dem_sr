import unittest

import torch

from dem_film_unet import (
    LOSS_PRESET_BASELINE,
    LOSS_PRESET_CHOICES,
    loss_dem_preset,
)
from losses.components import ElevationSmoothL1Loss, FlowDirectionProxyLoss, PitSpikePenaltyLoss, SlopeL1Loss
from losses.composite import CompositeLoss, build_composite_loss_from_config


class CompositeLossTests(unittest.TestCase):
    def test_elevation_component_zero_when_equal(self) -> None:
        comp = ElevationSmoothL1Loss()
        z_hat = torch.ones((2, 1, 4, 4))
        z_gt = torch.ones((2, 1, 4, 4))
        w = torch.ones((2, 1, 4, 4))
        value = comp(z_hat, z_gt, w)
        self.assertAlmostEqual(float(value), 0.0, places=7)

    def test_composite_loss_combines_components(self) -> None:
        loss_fn = CompositeLoss(
            [
                (ElevationSmoothL1Loss(), 1.0, True),
                (SlopeL1Loss(), 0.5, True),
            ]
        )
        outputs = {"z_hat": torch.zeros((1, 1, 4, 4), dtype=torch.float32)}
        batch = {
            "z_gt": torch.ones((1, 1, 4, 4), dtype=torch.float32),
            "w": torch.ones((1, 1, 4, 4), dtype=torch.float32),
        }
        bundle = loss_fn(outputs, batch)
        self.assertIn("elev", bundle.metrics)
        self.assertIn("slope", bundle.metrics)
        self.assertGreater(float(bundle.loss), 0.0)

    def test_preset_parity_against_legacy_loss_dem_preset(self) -> None:
        torch.manual_seed(7)
        z_hat = torch.randn((2, 1, 16, 16), dtype=torch.float32)
        z_gt = torch.randn((2, 1, 16, 16), dtype=torch.float32)
        w = torch.rand((2, 1, 16, 16), dtype=torch.float32) * 0.9 + 0.1

        cfg = {
            "lambda_slope": 0.5,
            "lambda_grad": 0.25,
            "lambda_curv": 0.1,
            "lambda_ms": 0.5,
            "lambda_sdf": 0.5,
            "lambda_contour": 0.25,
            "lambda_elev": 1.0,
            "contour_interval": 10.0,
            "smooth_l1_beta": 1.0,
        }
        legacy_cfg = {k: v for k, v in cfg.items() if k != "lambda_elev"}

        for preset in LOSS_PRESET_CHOICES:
            composite = build_composite_loss_from_config(cfg | {"loss_preset": preset})
            bundle = composite({"z_hat": z_hat}, {"z_gt": z_gt, "w": w})
            legacy_total, legacy_metrics = loss_dem_preset(z_hat, z_gt, w, preset=preset, **legacy_cfg)
            self.assertAlmostEqual(float(bundle.loss), float(legacy_total), places=6, msg=f"total mismatch for {preset}")
            for name, value in legacy_metrics.items():
                self.assertIn(name, bundle.metrics)
                self.assertAlmostEqual(bundle.metrics[name], float(value), places=6, msg=f"{name} mismatch for {preset}")

    def test_disable_component_override(self) -> None:
        cfg = {"loss_preset": LOSS_PRESET_BASELINE, "enable_slope": False}
        composite = build_composite_loss_from_config(cfg)
        z = torch.zeros((1, 1, 8, 8))
        bundle = composite({"z_hat": z}, {"z_gt": z, "w": torch.ones_like(z)})
        self.assertNotIn("slope", bundle.metrics)

    def test_hydrology_flow_component_zero_when_equal(self) -> None:
        comp = FlowDirectionProxyLoss()
        z = torch.randn((1, 1, 8, 8), dtype=torch.float32)
        w = torch.ones_like(z)
        value = comp(z, z, w, {"x_dem": torch.zeros((1, 5, 8, 8), dtype=torch.float32)})
        self.assertAlmostEqual(float(value), 0.0, places=6)

    def test_hydrology_pit_spike_component_penalizes_excess_extrema(self) -> None:
        comp = PitSpikePenaltyLoss(kernel_size=3)
        z_gt = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
        z_hat = z_gt.clone()
        z_hat[:, :, 4, 4] = 5.0
        w = torch.ones_like(z_gt)
        value = comp(z_hat, z_gt, w, {"x_dem": torch.zeros((1, 5, 8, 8), dtype=torch.float32)})
        self.assertGreater(float(value), 0.0)

    def test_hydrology_terms_can_be_enabled_in_composite(self) -> None:
        cfg = {
            "loss_preset": "baseline",
            "enable_hydro_flow": True,
            "enable_hydro_pit_spike": True,
            "lambda_hydro_flow": 0.01,
            "lambda_hydro_pit_spike": 0.005,
        }
        composite = build_composite_loss_from_config(cfg)
        z_hat = torch.randn((1, 1, 8, 8), dtype=torch.float32)
        z_gt = torch.randn((1, 1, 8, 8), dtype=torch.float32)
        batch = {
            "z_gt": z_gt,
            "w": torch.ones_like(z_gt),
            "x_dem": torch.zeros((1, 5, 8, 8), dtype=torch.float32),
        }
        bundle = composite({"z_hat": z_hat}, batch)
        self.assertIn("hydro_flow", bundle.metrics)
        self.assertIn("hydro_pit_spike", bundle.metrics)


if __name__ == "__main__":
    unittest.main()

