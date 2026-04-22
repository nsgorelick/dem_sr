# Plan: Mixture-of-Specialists with Terrain Router

## Goal
Use a routing model to send each sample (or pixel region) to specialized experts by terrain regime.

## Residual + Safeguard Requirements
- Every expert predicts a residual from `z_lr`; router combines residuals, not absolute DEMs.
- Reconstruct globally as `z_hat = z_lr + r_mix`.
- Apply residual clamping per-expert and/or after mixture aggregation.
- Keep weighted masking (`W`), finite-value sanitization, and trust/uncertainty conditioning across experts.
- Keep evaluation parity with baseline metrics and non-AU/AU split discipline.

## Why This Is Different
- Current setup is one global model for all terrain conditions.
- Specialists can focus on steep, hydrologic, urban, or flat regimes separately.

## Implementation Plan
1. Define strata using existing patch-table features (`p90_slope`, hydrology, uncertainty, etc.).
2. Train 2-4 specialist models on stratified subsets.
3. Train router to output expert weights from input features.
4. Infer with hard routing (single expert) or soft mixture.

## Required Beyond New Model
- **Input/metadata changes**
  - Ensure training loader can attach stratum metadata or derived routing features.
  - Optionally include region-level masks for spatial routing.
- **Training orchestration**
  - Multiple model trainings + router training stage.
  - Balanced sampling per stratum to avoid expert collapse.
- **Inference and serving**
  - Runtime dispatch logic to choose/load experts.
  - Model registry/versioning for expert set + router compatibility.
- **Evaluation updates**
  - Per-stratum metrics become first-class acceptance criteria.

## Data and Preprocessing Needs
- Need reliable stratum labels for all train/val stems.
- May need minimum-sample thresholds before enabling an expert.

## Evaluation Plan
- Compare global baseline vs MoE on each stratum and overall weighted metrics.
- Confirm router calibration (confidence vs realized error).

## Risks
- Operational complexity (many checkpoints).
- Little gain if routing features are weak/noisy.

## Pilot Exit Criteria
- Keep if worst-performing strata improve materially without hurting global RMSE.
