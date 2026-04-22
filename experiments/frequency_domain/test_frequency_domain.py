import unittest

import torch

from core.frequency import decompose_residual_laplacian, reconstruct_residual_from_bands
from experiments.frequency_domain import FrequencyDomainResidualModel


class FrequencyDomainTests(unittest.TestCase):
    def test_decomposition_reconstructs_residual(self) -> None:
        r = torch.randn((2, 1, 32, 32), dtype=torch.float32)
        low, mid, high = decompose_residual_laplacian(r)
        recon = reconstruct_residual_from_bands(low, mid, high)
        self.assertTrue(torch.allclose(recon, r, atol=1e-6, rtol=0.0))

    def test_model_outputs_bands_and_reconstruction(self) -> None:
        model = FrequencyDomainResidualModel()
        x_dem = torch.randn((1, 5, 32, 32), dtype=torch.float32)
        x_ae = torch.randn((1, 64, 32, 32), dtype=torch.float32)
        z_lr = torch.randn((1, 1, 32, 32), dtype=torch.float32)
        out = model(x_dem, x_ae, z_lr)
        self.assertIn("z_hat", out)
        self.assertIn("r_low_hat", out)
        self.assertIn("r_mid_hat", out)
        self.assertIn("r_high_hat", out)
        self.assertIn("r_hat", out)
        sum_hat = out["r_low_hat"] + out["r_mid_hat"] + out["r_high_hat"]
        capped = model.r_cap * torch.tanh(sum_hat / model.r_cap)
        self.assertTrue(torch.allclose(capped, out["r_hat"], atol=1e-6, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
