import tempfile
import unittest
from pathlib import Path

import torch

from core.checkpoints import (
    extract_model_state,
    load_checkpoint,
    make_training_checkpoint_payload,
    save_training_checkpoint,
)
from core.data_schema import validate_batch, validate_loss_outputs, validate_model_outputs
from core.metrics import (
    add_customer_example_fields,
    build_patch_table_context,
    compute_per_patch_metrics,
    compute_stratified_metrics,
    finalize_metric_sums,
    init_metric_sums,
    parse_patch_stem,
    set_contour_interval,
    update_metric_sums,
)
from experiments.registry import create_experiment, list_experiments


class CoreModulesTests(unittest.TestCase):
    def _make_batch(self, batch_size: int = 2, size: int = 4) -> dict[str, torch.Tensor]:
        shape = (batch_size, 1, size, size)
        return {
            "x_dem": torch.zeros(shape),
            "x_ae": torch.zeros(shape),
            "z_lr": torch.zeros(shape),
            "z_gt": torch.ones(shape),
            "w": torch.ones(shape),
        }

    def test_validate_batch_accepts_valid_batch(self) -> None:
        validate_batch(self._make_batch())

    def test_validate_batch_rejects_missing_key(self) -> None:
        batch = self._make_batch()
        del batch["x_ae"]
        with self.assertRaises(KeyError):
            validate_batch(batch)

    def test_validate_model_outputs_returns_z_hat(self) -> None:
        z_hat = torch.ones((1, 1, 2, 2))
        out = validate_model_outputs({"z_hat": z_hat})
        self.assertIs(out, z_hat)

    def test_validate_loss_outputs_normalizes_metrics(self) -> None:
        loss = torch.tensor(1.25)
        loss_out, metrics = validate_loss_outputs({"loss": loss, "metrics": {"elev": torch.tensor(2.0)}})
        self.assertIs(loss_out, loss)
        self.assertEqual(metrics["elev"], 2.0)

    def test_checkpoint_round_trip(self) -> None:
        model = torch.nn.Conv2d(1, 1, kernel_size=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        history = {"train_loss": [1.0], "val_loss": [2.0]}
        payload = make_training_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            data_root="/tmp/data",
            epoch=1,
            args={"arch": "film_unet"},
            history=history,
            train_size=10,
            val_size=2,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            save_training_checkpoint(path, payload)
            reloaded = load_checkpoint(path)
        self.assertIn("model", reloaded)
        self.assertEqual(reloaded["epoch"], 1)
        model_state = extract_model_state(reloaded)
        self.assertIn("weight", model_state)

    def test_metric_accumulators_produce_zero_error_when_equal(self) -> None:
        set_contour_interval(10.0)
        sums = init_metric_sums(torch.device("cpu"))
        pred = torch.ones((2, 1, 4, 4))
        z_gt = torch.ones((2, 1, 4, 4))
        w = torch.ones((2, 1, 4, 4))
        update_metric_sums(pred, z_gt, w, sums)
        metrics = finalize_metric_sums(sums, n_patches=2)
        self.assertAlmostEqual(metrics["elev_rmse_w"], 0.0, places=7)
        self.assertAlmostEqual(metrics["slope_rmse_w"], 0.0, places=7)

    def test_compute_per_patch_metrics_returns_patch_rows(self) -> None:
        pred = torch.zeros((2, 1, 4, 4))
        z_gt = torch.ones((2, 1, 4, 4))
        w = torch.ones((2, 1, 4, 4))
        per_patch = compute_per_patch_metrics(pred, z_gt, w)
        self.assertEqual(len(per_patch["elev_mae_w"]), 2)
        self.assertGreater(per_patch["elev_mae_w"][0], 0.0)

    def test_compute_stratified_metrics_groups_rows(self) -> None:
        rows = [
            {
                "slope_bin": "0-2",
                "hydrology_bin": "dry",
                "building_bin": "0",
                "uncertainty_bin": "q1",
                "model_sum_weights": 1.0,
                "model_elev_bias_w": 0.0,
                "model_elev_mae_w": 1.0,
                "model_elev_rmse_w": 1.0,
                "model_slope_mae_w": 1.0,
                "model_slope_rmse_w": 1.0,
                "model_slope_mae_deg_w": 1.0,
                "model_slope_rmse_deg_w": 1.0,
            },
            {
                "slope_bin": "0-2",
                "hydrology_bin": "dry",
                "building_bin": "0",
                "uncertainty_bin": "q1",
                "model_sum_weights": 1.0,
                "model_elev_bias_w": 0.0,
                "model_elev_mae_w": 3.0,
                "model_elev_rmse_w": 3.0,
                "model_slope_mae_w": 3.0,
                "model_slope_rmse_w": 3.0,
                "model_slope_mae_deg_w": 3.0,
                "model_slope_rmse_deg_w": 3.0,
            },
        ]
        out = compute_stratified_metrics(rows, ["model"])
        grouped = out["model"]["slope_bin"]["0-2"]
        self.assertEqual(grouped["n_patches"], 2.0)
        self.assertAlmostEqual(grouped["elev_mae_w"], 2.0, places=7)

    def test_parse_patch_stem(self) -> None:
        parsed = parse_patch_stem("12_34_56_AU_2020")
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["country"], "AU")
        self.assertIsNone(parse_patch_stem("bad_stem"))

    def test_customer_example_fields(self) -> None:
        row = {
            "z_lr_elev_mae_w": 2.0,
            "model_elev_mae_w": 1.0,
            "z_lr_elev_rmse_w": 2.0,
            "model_elev_rmse_w": 1.0,
            "z_lr_slope_mae_deg_w": 2.0,
            "model_slope_mae_deg_w": 1.0,
            "z_lr_slope_rmse_deg_w": 2.0,
            "model_slope_rmse_deg_w": 1.0,
        }
        add_customer_example_fields(row, baseline_source="z_lr", improved_source="model")
        self.assertIn("model_vs_z_lr_customer_example_score", row)
        self.assertGreater(float(row["model_vs_z_lr_customer_example_score"]), 0.0)

    def test_experiment_registry_baseline(self) -> None:
        self.assertIn("baseline", list_experiments())
        self.assertIn("hydrology", list_experiments())
        self.assertIn("two_stage", list_experiments())
        exp = create_experiment("baseline")
        self.assertEqual(exp.name, "baseline")
        hydrology = create_experiment("hydrology")
        self.assertEqual(hydrology.name, "hydrology")
        two_stage = create_experiment("two_stage")
        self.assertEqual(two_stage.name, "two_stage")

    def test_build_patch_table_context(self) -> None:
        patch_table = {
            "1_2_3_AU_2020": {
                "p90_slope": 3.0,
                "frac_shore": 0.0,
                "frac_water": 0.2,
                "has_edge": 0.0,
                "frac_building": 0.1,
                "mean_uncert": 5.0,
            }
        }
        context, meta = build_patch_table_context(patch_table, ["1_2_3_AU_2020", "4_5_6_AU_2020"])
        self.assertIn("1_2_3_AU_2020", context)
        self.assertIn("slope_bin", context["1_2_3_AU_2020"])
        self.assertIn("strata_fields", meta)


if __name__ == "__main__":
    unittest.main()

