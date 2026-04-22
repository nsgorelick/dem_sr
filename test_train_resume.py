import argparse
import tempfile
import unittest
from pathlib import Path

import torch

from core.checkpoints import make_training_checkpoint_payload, save_training_checkpoint
from train_experiment import _resume_from_checkpoint


class TrainResumeTests(unittest.TestCase):
    def _build_args(self, **overrides: object) -> argparse.Namespace:
        base = {
            "arch": "film_unet",
            "epochs": 5,
        }
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_resume_restores_state_and_epoch(self) -> None:
        model = torch.nn.Conv2d(1, 1, kernel_size=1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "resume.pt"
            payload = make_training_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                data_root="/tmp/data",
                epoch=2,
                args={"arch": "film_unet"},
                history={"train_loss": [1.0, 0.9], "epoch_seconds": [1.0, 1.2], "val_loss": []},
                train_size=10,
                val_size=0,
            )
            save_training_checkpoint(ckpt_path, payload)
            history = {"train_loss": [], "epoch_seconds": []}
            start_epoch, out_history = _resume_from_checkpoint(
                resume_path=ckpt_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                args=self._build_args(epochs=4),
                history=history,
            )
        self.assertEqual(start_epoch, 2)
        self.assertEqual(len(out_history["train_loss"]), 2)

    def test_resume_rejects_arch_mismatch(self) -> None:
        model = torch.nn.Conv2d(1, 1, kernel_size=1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "resume.pt"
            payload = make_training_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                data_root="/tmp/data",
                epoch=1,
                args={"arch": "other_arch"},
                history={"train_loss": [1.0], "epoch_seconds": [1.0], "val_loss": []},
                train_size=10,
                val_size=0,
            )
            save_training_checkpoint(ckpt_path, payload)
            with self.assertRaises(ValueError):
                _resume_from_checkpoint(
                    resume_path=ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    args=self._build_args(arch="film_unet"),
                    history={"train_loss": [], "epoch_seconds": []},
                )


if __name__ == "__main__":
    unittest.main()

