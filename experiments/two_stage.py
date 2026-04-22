"""Two-stage residual experiment for global + local scale separation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from core.checkpoints import extract_model_state, load_checkpoint
from dem_film_unet import ARCH_FILM, LOSS_PRESET_BASELINE
from experiments.base import Experiment, LossBundle
from losses.composite import build_composite_loss_from_config
from models.wrappers.factory import create_experiment_model


class TwoStageResidualModel(torch.nn.Module):
    """Wrap two core models into a staged residual pipeline."""

    def __init__(
        self,
        *,
        arch: str,
        coarse_pool_kernel: int,
        train_stage: str,
        stage_a_checkpoint: str | None = None,
    ) -> None:
        super().__init__()
        self.stage_a = create_experiment_model(arch)
        self.stage_b = create_experiment_model(arch)
        self.coarse_pool_kernel = max(int(coarse_pool_kernel), 1)
        self.train_stage = str(train_stage).lower()
        if self.train_stage not in {"stage_a", "stage_b"}:
            raise ValueError("two_stage_train_stage must be one of: stage_a, stage_b")
        if self.train_stage == "stage_b":
            if not stage_a_checkpoint:
                raise ValueError("stage_b training requires --two-stage-a-checkpoint")
            checkpoint = load_checkpoint(Path(stage_a_checkpoint))
            stage_a_state = extract_model_state(checkpoint)
            self._load_prefixed_state(self.stage_a, stage_a_state, prefix="stage_a.")
            for param in self.stage_a.parameters():
                param.requires_grad = False

    @staticmethod
    def _load_prefixed_state(
        module: torch.nn.Module,
        state: dict[str, Any],
        *,
        prefix: str,
    ) -> None:
        if any(key.startswith(prefix) for key in state):
            sub_state = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
            module.load_state_dict(sub_state, strict=True)
        else:
            module.load_state_dict(state, strict=True)

    def _coarsen_residual(self, residual: torch.Tensor) -> torch.Tensor:
        if self.coarse_pool_kernel <= 1:
            return residual
        pooled = F.avg_pool2d(residual, kernel_size=self.coarse_pool_kernel, stride=self.coarse_pool_kernel)
        return F.interpolate(pooled, size=residual.shape[-2:], mode="bilinear", align_corners=False)

    def forward(
        self,
        x_dem: torch.Tensor,
        x_ae: torch.Tensor,
        z_lr: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        z_a_raw = self.stage_a(x_dem, x_ae, z_lr)
        r_coarse = self._coarsen_residual(z_a_raw - z_lr)
        z_mid = z_lr + r_coarse
        z_hat = z_mid
        if self.train_stage == "stage_b" or not self.training:
            # Stage B can override Stage A via signed residual around z_mid.
            z_hat = self.stage_b(x_dem, x_ae, z_mid)
        return {"z_hat": z_hat, "z_stage_a": z_mid, "r_coarse": r_coarse, "r_detail": z_hat - z_mid}

    def trainable_parameters(self):
        if self.train_stage == "stage_b":
            return self.stage_b.parameters()
        return self.stage_a.parameters()


class TwoStageResidualExperiment(Experiment):
    name = "two_stage"

    def build_model(self, cfg: dict[str, Any]) -> torch.nn.Module:
        arch = str(cfg.get("arch", ARCH_FILM))
        return TwoStageResidualModel(
            arch=arch,
            coarse_pool_kernel=int(cfg.get("two_stage_coarse_pool_kernel", 4)),
            train_stage=str(cfg.get("two_stage_train_stage", "stage_a")),
            stage_a_checkpoint=cfg.get("two_stage_a_checkpoint"),
        )

    def model_forward(self, model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        outputs = model(batch["x_dem"], batch["x_ae"], batch["z_lr"])
        return outputs

    def build_loss(self, cfg: dict[str, Any]):
        loss_cfg = dict(cfg)
        loss_cfg.setdefault("loss_preset", LOSS_PRESET_BASELINE)
        composite = build_composite_loss_from_config(loss_cfg)

        def _loss_fn(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> LossBundle:
            bundle = composite(outputs, batch)
            metrics = dict(bundle.metrics)
            if "z_stage_a" in outputs:
                stage_a_bundle = composite({"z_hat": outputs["z_stage_a"]}, batch)
                metrics["stage_a_total"] = float(stage_a_bundle.loss.detach())
            return LossBundle(loss=bundle.loss, metrics=metrics)

        return _loss_fn

