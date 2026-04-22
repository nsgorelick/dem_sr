import unittest

import eval_experiment
import pretrain_experiment
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

    def test_train_parser_accepts_pretrained_encoder_checkpoint(self) -> None:
        parser = train_experiment.build_parser()
        args = parser.parse_args(["--pretrained-encoder-checkpoint", "runs/pretrain/encoder_pretrain.pt"])
        self.assertEqual(str(args.pretrained_encoder_checkpoint), "runs/pretrain/encoder_pretrain.pt")

    def test_train_parser_accepts_two_stage_args(self) -> None:
        parser = train_experiment.build_parser()
        args = parser.parse_args(
            [
                "--experiment",
                "two_stage",
                "--two-stage-train-stage",
                "stage_b",
                "--two-stage-a-checkpoint",
                "stage_a.pt",
            ]
        )
        self.assertEqual(args.two_stage_train_stage, "stage_b")
        self.assertEqual(str(args.two_stage_a_checkpoint), "stage_a.pt")

    def test_train_parser_accepts_mos_args(self) -> None:
        parser = train_experiment.build_parser()
        args = parser.parse_args(
            [
                "--experiment",
                "mixture_specialists",
                "--mos-num-experts",
                "3",
                "--mos-router-temperature",
                "0.8",
            ]
        )
        self.assertEqual(args.mos_num_experts, 3)
        self.assertEqual(args.mos_router_temperature, 0.8)

    def test_train_parser_accepts_frequency_domain_band_args(self) -> None:
        parser = train_experiment.build_parser()
        args = parser.parse_args(
            [
                "--experiment",
                "frequency_domain",
                "--lambda-band-low",
                "0.6",
                "--lambda-band-high-tv",
                "0.3",
            ]
        )
        self.assertEqual(args.lambda_band_low, 0.6)
        self.assertEqual(args.lambda_band_high_tv, 0.3)

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
        args = parser.parse_args(["--prediction-source", "model", "stage_a", "z_lr", "--list-from-root"])
        self.assertEqual(args.prediction_source, ["model", "stage_a", "z_lr"])

    def test_pretrain_parser_accepts_masked_reconstruction_args(self) -> None:
        parser = pretrain_experiment.build_parser()
        args = parser.parse_args(
            [
                "--mask-ratio-dem",
                "0.5",
                "--mask-ratio-ae",
                "0.3",
                "--checkpoint-out",
                "runs/pretrain/encoder.pt",
            ]
        )
        self.assertEqual(args.mask_ratio_dem, 0.5)
        self.assertEqual(args.mask_ratio_ae, 0.3)

    def test_eval_parser_accepts_sliding_window_args(self) -> None:
        parser = eval_experiment.build_parser()
        args = parser.parse_args(
            [
                "--list-from-root",
                "--sliding-window-tile-size",
                "512",
                "--sliding-window-overlap",
                "64",
            ]
        )
        self.assertEqual(args.sliding_window_tile_size, 512)
        self.assertEqual(args.sliding_window_overlap, 64)


if __name__ == "__main__":
    unittest.main()

