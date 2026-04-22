import argparse
import unittest
from pathlib import Path

from core.config import (
    ExperimentConfig,
    add_shared_experiment_args,
    apply_namespace_preset_defaults,
    apply_preset_defaults,
    config_to_dict,
    resolve_config,
)


class ExperimentConfigTests(unittest.TestCase):
    def test_resolve_config_uses_namespace_values(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--experiment", default="baseline")
        parser.add_argument("--list-from-root", action="store_true")
        add_shared_experiment_args(parser)
        args = parser.parse_args(
            [
                "--experiment",
                "baseline",
                "--data-root",
                "/tmp/data",
                "--manifest",
                "manifest.txt",
                "--batch-size",
                "12",
                "--workers",
                "4",
                "--amp",
                "--precomputed-weight",
                "--contour-interval",
                "20",
                "--arch",
                "film_unet",
                "--loss-preset",
                "baseline",
            ]
        )
        cfg = resolve_config(args)
        self.assertEqual(cfg.data_root, "/tmp/data")
        self.assertEqual(cfg.manifest, Path("manifest.txt"))
        self.assertEqual(cfg.batch_size, 12)
        self.assertTrue(cfg.amp)
        self.assertTrue(cfg.precomputed_weight)
        self.assertEqual(cfg.contour_interval, 20.0)

    def test_resolve_config_falls_back_to_default_root(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--experiment", default="baseline")
        parser.add_argument("--list-from-root", action="store_true")
        add_shared_experiment_args(parser)
        args = parser.parse_args([])
        args.data_root = None
        cfg = resolve_config(args, default_data_root="/checkpoint/root")
        self.assertEqual(cfg.data_root, "/checkpoint/root")

    def test_config_to_dict_includes_expected_keys(self) -> None:
        cfg = ExperimentConfig(data_root="/tmp")
        out = config_to_dict(cfg)
        self.assertEqual(out["data_root"], "/tmp")
        self.assertIn("loss_preset", out)
        self.assertIn("contour_interval", out)

    def test_apply_preset_defaults_applies_on_default_values(self) -> None:
        cfg = ExperimentConfig(batch_size=32, workers=2)
        out = apply_preset_defaults(cfg, {"batch_size": 4, "workers": 0})
        self.assertEqual(out.batch_size, 32)
        self.assertEqual(out.workers, 0)

    def test_apply_namespace_preset_defaults_uses_parser_defaults(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch-size", type=int, default=4)
        parser.add_argument("--workers", type=int, default=2)
        parser.add_argument("--loss-system", default="preset")

        args_default = parser.parse_args([])
        apply_namespace_preset_defaults(args_default, parser, {"batch_size": 16, "loss_system": "composite"})
        self.assertEqual(args_default.batch_size, 16)
        self.assertEqual(args_default.loss_system, "composite")

        args_override = parser.parse_args(["--batch-size", "32"])
        apply_namespace_preset_defaults(args_override, parser, {"batch_size": 16})
        self.assertEqual(args_override.batch_size, 32)


if __name__ == "__main__":
    unittest.main()

