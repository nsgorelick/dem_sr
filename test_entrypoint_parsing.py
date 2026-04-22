import unittest

import eval_experiment
import train_experiment
from core.config import apply_namespace_preset_defaults
from experiments.config_presets import get_preset


class EntrypointParsingTests(unittest.TestCase):
    def test_train_parser_accepts_composite_loss_args(self) -> None:
        parser = train_experiment.build_parser()
        args = parser.parse_args(
            [
                "--experiment",
                "baseline",
                "--preset",
                "plan09-composite",
            ]
        )
        apply_namespace_preset_defaults(args, parser, get_preset("train", args.preset))
        self.assertEqual(args.loss_system, "composite")
        self.assertEqual(args.lambda_elev, 1.0)

    def test_train_parser_cli_override_beats_preset(self) -> None:
        parser = train_experiment.build_parser()
        args = parser.parse_args(
            [
                "--preset",
                "smoke",
                "--batch-size",
                "12",
            ]
        )
        apply_namespace_preset_defaults(args, parser, get_preset("train", args.preset))
        self.assertEqual(args.batch_size, 12)

    def test_eval_parser_accepts_preset(self) -> None:
        parser = eval_experiment.build_parser()
        args = parser.parse_args(["--preset", "smoke", "--list-from-root"])
        apply_namespace_preset_defaults(args, parser, get_preset("eval", args.preset))
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.workers, 0)

    def test_eval_parser_accepts_prediction_sources(self) -> None:
        parser = eval_experiment.build_parser()
        args = parser.parse_args(["--prediction-source", "model", "z_lr", "--list-from-root"])
        self.assertEqual(args.prediction_source, ["model", "z_lr"])


if __name__ == "__main__":
    unittest.main()

