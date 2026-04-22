import unittest

import torch

from experiments.mixture_specialists import SharedBackboneMixtureUNet


class MixtureSpecialistsTests(unittest.TestCase):
    def test_forward_produces_soft_weights_and_prediction(self) -> None:
        model = SharedBackboneMixtureUNet(num_experts=3, router_temperature=1.0)
        x_dem = torch.randn((2, 5, 32, 32))
        x_ae = torch.randn((2, 64, 32, 32))
        z_lr = torch.randn((2, 1, 32, 32))
        out = model(x_dem, x_ae, z_lr)
        self.assertIn("z_hat", out)
        self.assertIn("specialist_weights", out)
        self.assertIn("expert_residuals", out)
        self.assertEqual(tuple(out["z_hat"].shape), (2, 1, 32, 32))
        self.assertEqual(tuple(out["specialist_weights"].shape), (2, 3))
        sums = out["specialist_weights"].sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-6))
        self.assertTrue(torch.all(out["specialist_weights"] > 0.0))

    def test_expert_outputs_are_not_identical(self) -> None:
        torch.manual_seed(7)
        model = SharedBackboneMixtureUNet(num_experts=3, router_temperature=1.0)
        x_dem = torch.randn((1, 5, 32, 32))
        x_ae = torch.randn((1, 64, 32, 32))
        z_lr = torch.randn((1, 1, 32, 32))
        out = model(x_dem, x_ae, z_lr)
        experts = out["expert_residuals"]
        d01 = (experts[:, 0] - experts[:, 1]).abs().mean()
        d12 = (experts[:, 1] - experts[:, 2]).abs().mean()
        self.assertGreater(float(d01 + d12), 0.0)


if __name__ == "__main__":
    unittest.main()

