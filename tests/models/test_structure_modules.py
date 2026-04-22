import unittest

from core.io import STACK_BAND_NAMES, STACK_BAND_TO_INDEX
from eval.stratify import STRATA_FIELDS
from models.wrappers.factory import create_experiment_model


class StructureModulesTests(unittest.TestCase):
    def test_stack_spec_indices(self) -> None:
        self.assertEqual(STACK_BAND_NAMES[0], "z_gt10")
        self.assertEqual(STACK_BAND_TO_INDEX["weight"], 9)

    def test_eval_stratify_exports(self) -> None:
        self.assertIn("slope_bin", STRATA_FIELDS)

    def test_model_wrapper_builds_baseline_arch(self) -> None:
        model = create_experiment_model("film_unet")
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()

