import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from experiments.base import LossBundle
from train.engine import run_epoch


class _TinyDataset(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        base = torch.full((1, 2, 2), float(idx) / 10.0)
        return {
            "x_dem": base.clone(),
            "x_ae": base.clone(),
            "z_lr": base.clone(),
            "z_gt": base.clone() + 1.0,
            "w": torch.ones((1, 2, 2)),
        }


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key in batch[0]:
        out[key] = torch.stack([sample[key] for sample in batch], dim=0)
    return out


def _forward(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    stacked = torch.cat((batch["x_dem"], batch["x_ae"], batch["z_lr"]), dim=1)
    return {"z_hat": model(stacked)}


def _loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> LossBundle:
    err = outputs["z_hat"] - batch["z_gt"]
    loss = (err * err).mean()
    return LossBundle(loss=loss, metrics={"mse": float(loss.detach())})


class TrainEngineTests(unittest.TestCase):
    def test_run_epoch_train_updates_model(self) -> None:
        model = torch.nn.Conv2d(3, 1, kernel_size=1, bias=False)
        init_weight = model.weight.detach().clone()
        loader = DataLoader(_TinyDataset(), batch_size=2, shuffle=False, collate_fn=_collate)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        metrics = run_epoch(
            model=model,
            loader=loader,
            device=torch.device("cpu"),
            model_forward=_forward,
            loss_fn=_loss,
            optimizer=optimizer,
            scaler=None,
            amp_enabled=False,
            train=True,
        )

        self.assertIn("loss", metrics)
        self.assertGreater(metrics["n_batches"], 0.0)
        self.assertFalse(torch.equal(init_weight, model.weight.detach()))

    def test_run_epoch_eval_does_not_require_optimizer(self) -> None:
        model = torch.nn.Conv2d(3, 1, kernel_size=1, bias=False)
        loader = DataLoader(_TinyDataset(), batch_size=2, shuffle=False, collate_fn=_collate)
        with torch.no_grad():
            metrics = run_epoch(
                model=model,
                loader=loader,
                device=torch.device("cpu"),
                model_forward=_forward,
                loss_fn=_loss,
                optimizer=None,
                scaler=None,
                amp_enabled=False,
                train=False,
            )
        self.assertIn("mse", metrics)


if __name__ == "__main__":
    unittest.main()

