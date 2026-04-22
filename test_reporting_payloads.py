import unittest
from pathlib import Path

from core.reporting import build_eval_payload, build_train_payload


class ReportingPayloadTests(unittest.TestCase):
    def test_build_train_payload(self) -> None:
        payload = build_train_payload(
            experiment="baseline",
            checkpoint_out="ckpt.pt",
            data_root="/data/training",
            epochs=3,
            history={"train_loss": [1.0], "epoch_seconds": [2.0]},
            train_size=100,
            config={"batch_size": 4, "manifest": Path("manifest.txt")},
        )
        self.assertEqual(payload["kind"], "train")
        self.assertEqual(payload["epochs"], 3)
        self.assertEqual(payload["train_size"], 100)
        self.assertEqual(payload["config"]["manifest"], "manifest.txt")

    def test_build_eval_payload(self) -> None:
        payload = build_eval_payload(
            experiment="baseline",
            prediction_sources=["model", "z_lr"],
            checkpoint="ckpt.pt",
            data_root="/data/training",
            manifest="manifest.txt",
            list_from_root=False,
            contour_interval_m=10.0,
            metrics_by_source={"model": {"elev_rmse_w": 1.0}},
            config={"batch_size": 8},
        )
        self.assertEqual(payload["kind"], "eval")
        self.assertEqual(payload["prediction_source"], ["model", "z_lr"])
        self.assertIn("metrics_by_source", payload)


if __name__ == "__main__":
    unittest.main()

