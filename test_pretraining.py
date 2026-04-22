import tempfile
import unittest
from pathlib import Path

import torch

from core.pretraining import extract_encoder_state_dict, load_pretrained_encoder
from dem_film_unet import create_model


class PretrainingUtilsTests(unittest.TestCase):
    def test_extract_encoder_state_contains_expected_prefixes(self) -> None:
        model = create_model("film_unet")
        state = extract_encoder_state_dict(model)
        self.assertTrue(any(key.startswith("dem_b0.") for key in state))
        self.assertTrue(any(key.startswith("ae_b0.") for key in state))

    def test_load_pretrained_encoder_from_encoder_state(self) -> None:
        source = create_model("film_unet")
        target = create_model("film_unet")
        source_state = extract_encoder_state_dict(source)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ssl.pt"
            torch.save({"encoder_state": source_state}, path)
            loaded, skipped = load_pretrained_encoder(target, path)
        self.assertGreater(loaded, 0)
        self.assertEqual(skipped, 0)


if __name__ == "__main__":
    unittest.main()

