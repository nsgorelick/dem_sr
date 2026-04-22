import tempfile
import unittest
from pathlib import Path

from core.run_config import (
    load_run_config,
    resolve_description,
    section_defaults,
    standardized_eval_output_path,
)


class RunConfigTests(unittest.TestCase):
    def test_load_and_section_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "cfg.json"
            p.write_text(
                """
{
  "description": "demo",
  "shared": {"batch_size": 4, "workers": 2},
  "train": {"epochs": 3}
}
""".strip(),
                encoding="utf-8",
            )
            cfg = load_run_config(p)
        defaults = section_defaults(cfg, "train")
        self.assertEqual(defaults["batch_size"], 4)
        self.assertEqual(defaults["epochs"], 3)

    def test_resolve_description(self) -> None:
        cfg = {"description": "my run"}
        out = resolve_description(cfg, Path("foo.json"), None)
        self.assertEqual(out, "my run")
        out2 = resolve_description({}, Path("foo.json"), None)
        self.assertEqual(out2, "foo")

    def test_standardized_eval_output_path(self) -> None:
        p = standardized_eval_output_path(config_path=Path("configs/base.json"), description="My Test Run")
        self.assertEqual(p.as_posix(), "configs/results/base_eval_results_my-test-run.json")


if __name__ == "__main__":
    unittest.main()

