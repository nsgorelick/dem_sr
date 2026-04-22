import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from eval.engine import run_eval_epoch, run_eval_epoch_multi_source, run_eval_epoch_multi_source_with_rows


class _EvalDataset(Dataset):
    def __len__(self) -> int:
        return 6

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        base = torch.full((1, 2, 2), float(idx) / 10.0)
        return {
            "x_dem": base.clone(),
            "x_ae": base.clone(),
            "z_lr": base.clone(),
            "z_gt": base.clone() + 1.0,
            "w": torch.ones((1, 2, 2)),
        }


class _LargeEvalDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        base = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8) + float(idx)
        return {
            "x_dem": base.clone(),
            "x_ae": base.repeat(64, 1, 1) / 100.0,
            "z_lr": base.clone(),
            "z_gt": base.clone(),
            "w": torch.ones((1, 8, 8)),
        }


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key in batch[0]:
        out[key] = torch.stack([sample[key] for sample in batch], dim=0)
    return out


def _forward(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    stacked = torch.cat((batch["x_dem"], batch["x_ae"], batch["z_lr"]), dim=1)
    return {"z_hat": model(stacked)}


def _forward_identity(
    model: torch.nn.Module,  # noqa: ARG001
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {"z_hat": batch["z_lr"]}


class EvalEngineTests(unittest.TestCase):
    def test_eval_engine_returns_metrics(self) -> None:
        model = torch.nn.Conv2d(3, 1, kernel_size=1, bias=False)
        loader = DataLoader(_EvalDataset(), batch_size=2, shuffle=False, collate_fn=_collate)
        metrics = run_eval_epoch(
            model=model,
            loader=loader,
            device=torch.device("cpu"),
            model_forward=_forward,
            amp_enabled=False,
        )
        self.assertIn("elev_rmse_w", metrics)
        self.assertIn("slope_rmse_w", metrics)
        self.assertEqual(metrics["n_patches"], 6.0)

    def test_eval_engine_multi_source_includes_zlr(self) -> None:
        model = torch.nn.Conv2d(3, 1, kernel_size=1, bias=False)
        loader = DataLoader(_EvalDataset(), batch_size=2, shuffle=False, collate_fn=_collate)
        by_source = run_eval_epoch_multi_source(
            model=model,
            loader=loader,
            device=torch.device("cpu"),
            model_forward=_forward,
            amp_enabled=False,
            prediction_sources=["model", "z_lr"],
        )
        self.assertIn("model", by_source)
        self.assertIn("z_lr", by_source)
        self.assertEqual(by_source["model"]["n_patches"], 6.0)

    def test_eval_engine_multi_source_with_rows(self) -> None:
        model = torch.nn.Conv2d(3, 1, kernel_size=1, bias=False)
        loader = DataLoader(_EvalDataset(), batch_size=2, shuffle=False, collate_fn=_collate)
        by_source, rows = run_eval_epoch_multi_source_with_rows(
            model=model,
            loader=loader,
            device=torch.device("cpu"),
            model_forward=_forward,
            amp_enabled=False,
            prediction_sources=["model", "z_lr"],
        )
        self.assertIn("model", by_source)
        self.assertEqual(len(rows), 6)
        self.assertIn("model_elev_rmse_w", rows[0])
        self.assertIn("z_lr_elev_rmse_w", rows[0])

    def test_eval_engine_sliding_window_model_is_seam_free_for_identity(self) -> None:
        loader = DataLoader(_LargeEvalDataset(), batch_size=2, shuffle=False, collate_fn=_collate)
        by_source = run_eval_epoch_multi_source(
            model=torch.nn.Identity(),
            loader=loader,
            device=torch.device("cpu"),
            model_forward=_forward_identity,
            amp_enabled=False,
            prediction_sources=["model", "z_lr"],
            sliding_window_tile_size=4,
            sliding_window_overlap=2,
        )
        self.assertAlmostEqual(by_source["model"]["elev_rmse_w"], 0.0, places=5)
        self.assertAlmostEqual(by_source["model"]["elev_rmse_w"], by_source["z_lr"]["elev_rmse_w"], places=5)


if __name__ == "__main__":
    unittest.main()

