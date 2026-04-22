import unittest

from core.config import ExperimentConfig, apply_preset_defaults
from experiments.config_presets import get_preset, list_presets


class ConfigPresetTests(unittest.TestCase):
    def test_list_presets_contains_baseline(self) -> None:
        self.assertIn("baseline", list_presets("train"))
        self.assertIn("baseline", list_presets("eval"))

    def test_get_preset_smoke_train(self) -> None:
        preset = get_preset("train", "smoke")
        self.assertEqual(preset["max_patches"], 16)
        self.assertEqual(preset["workers"], 0)

    def test_get_preset_plan09_composite(self) -> None:
        preset = get_preset("train", "plan09-composite")
        self.assertEqual(preset["loss_system"], "composite")
        self.assertIn("lambda_elev", preset)
        self.assertIn("lambda_slope", preset)

    def test_apply_preset_only_changes_defaults(self) -> None:
        cfg = ExperimentConfig(batch_size=32, workers=2, max_patches=None)
        preset = {"batch_size": 4, "workers": 0, "max_patches": 64}
        out = apply_preset_defaults(cfg, preset)
        self.assertEqual(out.batch_size, 32)
        self.assertEqual(out.workers, 0)
        self.assertEqual(out.max_patches, 64)


if __name__ == "__main__":
    unittest.main()

