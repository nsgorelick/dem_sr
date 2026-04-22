import tempfile
import unittest
from pathlib import Path

import torch

from experiments.two_stage import TwoStageResidualModel


class _StageAPlusTwo(torch.nn.Module):
    def forward(self, x_dem: torch.Tensor, x_ae: torch.Tensor, z_lr: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return z_lr + 2.0


class _StageBMinusTwo(torch.nn.Module):
    def forward(self, x_dem: torch.Tensor, x_ae: torch.Tensor, z_lr: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return z_lr - 2.0


class TwoStageExperimentTests(unittest.TestCase):
    def _batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_lr = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
        x_dem = torch.zeros((1, 5, 8, 8), dtype=torch.float32)
        x_ae = torch.zeros((1, 64, 8, 8), dtype=torch.float32)
        return x_dem, x_ae, z_lr

    def test_stage_a_mode_outputs_stage_a_only(self) -> None:
        model = TwoStageResidualModel(arch="film_unet", coarse_pool_kernel=1, train_stage="stage_a")
        model.stage_a = _StageAPlusTwo()
        model.stage_b = _StageBMinusTwo()
        x_dem, x_ae, z_lr = self._batch()
        out = model(x_dem, x_ae, z_lr)
        self.assertTrue(torch.allclose(out["z_stage_a"], z_lr + 2.0))
        self.assertTrue(torch.allclose(out["z_hat"], z_lr + 2.0))

    def test_stage_b_mode_can_override_stage_a_mistake(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "stage_a.pt"
            baseline = TwoStageResidualModel(arch="film_unet", coarse_pool_kernel=1, train_stage="stage_a")
            torch.save({"model": baseline.state_dict()}, ckpt)
            model = TwoStageResidualModel(
                arch="film_unet",
                coarse_pool_kernel=1,
                train_stage="stage_b",
                stage_a_checkpoint=str(ckpt),
            )
        model.stage_a = _StageAPlusTwo()
        model.stage_b = _StageBMinusTwo()
        x_dem, x_ae, z_lr = self._batch()
        out = model(x_dem, x_ae, z_lr)
        self.assertTrue(torch.allclose(out["z_stage_a"], z_lr + 2.0))
        self.assertTrue(torch.allclose(out["z_hat"], z_lr))

    def test_stage_b_training_freezes_stage_a_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "stage_a.pt"
            baseline = TwoStageResidualModel(arch="film_unet", coarse_pool_kernel=1, train_stage="stage_a")
            torch.save({"model": baseline.state_dict()}, ckpt)
            model = TwoStageResidualModel(
                arch="film_unet",
                coarse_pool_kernel=1,
                train_stage="stage_b",
                stage_a_checkpoint=str(ckpt),
            )
        self.assertTrue(all(not p.requires_grad for p in model.stage_a.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.stage_b.parameters()))


if __name__ == "__main__":
    unittest.main()
