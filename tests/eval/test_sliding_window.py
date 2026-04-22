import unittest

import torch

from eval.sliding_window import predict_model_sliding_window


def _identity_forward(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:  # noqa: ARG001
    return {"z_hat": batch["z_lr"]}


class SlidingWindowInferenceTests(unittest.TestCase):
    def test_stitching_matches_full_prediction_for_identity(self) -> None:
        z = torch.arange(18 * 20, dtype=torch.float32).reshape(1, 1, 18, 20)
        batch = {
            "x_dem": z.repeat(1, 5, 1, 1),
            "x_ae": z.repeat(1, 64, 1, 1),
            "z_lr": z,
        }
        pred = predict_model_sliding_window(
            model=torch.nn.Identity(),
            batch=batch,
            model_forward=_identity_forward,
            tile_size=7,
            overlap=3,
            amp_enabled=False,
        )
        self.assertTrue(torch.allclose(pred, z, atol=1e-4, rtol=0.0))

    def test_invalid_overlap_raises(self) -> None:
        z = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
        batch = {"x_dem": z.repeat(1, 5, 1, 1), "x_ae": z.repeat(1, 64, 1, 1), "z_lr": z}
        with self.assertRaises(ValueError):
            predict_model_sliding_window(
                model=torch.nn.Identity(),
                batch=batch,
                model_forward=_identity_forward,
                tile_size=4,
                overlap=4,
                amp_enabled=False,
            )


if __name__ == "__main__":
    unittest.main()

