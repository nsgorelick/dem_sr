import unittest

from experiments.registry import create_experiment, list_experiments


class ExperimentRegistryTests(unittest.TestCase):
    def test_experiment_registry_baseline(self) -> None:
        self.assertIn("baseline", list_experiments())
        self.assertIn("frequency_domain", list_experiments())
        self.assertIn("hydrology", list_experiments())
        self.assertIn("mixture_specialists", list_experiments())
        self.assertIn("two_stage", list_experiments())
        self.assertEqual(create_experiment("baseline").name, "baseline")
        self.assertEqual(create_experiment("frequency_domain").name, "frequency_domain")
        self.assertEqual(create_experiment("hydrology").name, "hydrology")
        self.assertEqual(create_experiment("mixture_specialists").name, "mixture_specialists")
        self.assertEqual(create_experiment("two_stage").name, "two_stage")


if __name__ == "__main__":
    unittest.main()
