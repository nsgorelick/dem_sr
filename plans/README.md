# DEM Alternative Approaches: Prioritized Execution Plan

This document ranks the plans in this folder by expected upside, implementation effort, and time-to-signal under the current non-AU train / AU validation protocol.

## Cross-Cutting Constraints (Apply to Every Plan)

- Predict a **residual** and reconstruct with `z_hat = z_lr + r` (not direct absolute DEM prediction).
- Preserve existing **residual safeguards**:
  - residual clamping/capping (`r_cap` style bound),
  - weighted loss mask `W` / valid-mask handling,
  - finite-value sanitization and nodata exclusion,
  - trust/uncertainty/mask channels retained as conditioning.
- Preserve existing **evaluation safeguards**:
  - compare against `z_lr` baseline on identical manifests,
  - report elevation + slope + gradient + Laplacian + SDF metrics,
  - enforce non-AU train / AU validation split discipline,
  - require full AU re-check (`VAL_HARD_FRACTION=1.0`) before promotion.

## Quick Ranking (Effort vs Upside)

| Rank | Approach | Effort | Expected Upside | Time to First Signal | Notes |
|---|---|---|---|---|---|
| 1 | Two-stage global+local | Medium | High | Fast | Strong fit for bias + detail decomposition |
| 2 | Large-context tile training | Medium-High | High | Medium | Likely to improve context-limited terrain cases |
| 3 | Mixture-of-specialists | Medium-High | Medium-High | Medium | Leverages existing strata/eval bins |
| 4 | Frequency-domain learning | Medium | Medium-High | Medium | Targets slope/curvature behavior directly |
| 5 | Self-supervised pretraining | High | Medium-High | Slow-Medium | Can help AU transfer if unlabeled corpus is large |
| 6 | Variational unrolling | High | Medium | Medium-Slow | Strong priors, but optimization is tricky |
| 7 | Diffusion refinement | Very High | Very High (but uncertain) | Slow | Highest upside, highest compute/complexity |
| 8 | Non-neural baselines | Low-Medium | Diagnostic Medium | Fast | Good control baseline and failure-mode detector |

## Recommended Execution Order

1. `02_two_stage_global_local.md`
2. `05_large_context_tiles.md`
3. `06_mixture_of_specialists.md`
4. `03_frequency_domain.md`
5. `08_non_neural_baselines.md` (run early in parallel if possible)
6. `07_self_supervised_pretraining.md`
7. `04_variational_unrolling.md`
8. `01_diffusion_refinement.md`

Rationale:
- Start with methods that keep most of the current stack intact but change problem structure.
- Add one architectural "shape" change at a time before taking on major infrastructure-heavy research.
- Keep a non-neural baseline running in parallel to identify whether deep modeling is the true bottleneck.

## Suggested Program Phases

## Phase 1: Fast Structural Pivots (2-3 weeks)
- Two-stage global+local
- Large-context tiles
- Mixture-of-specialists (small number of experts)
- Non-neural baseline in parallel

Deliverable:
- Pick top 1-2 methods by AU full-val metrics and stratified stability.

## Phase 2: Representation and Frequency Methods (2-3 weeks)
- Frequency-domain model
- Self-supervised pretraining (small objective sweep)

Deliverable:
- Determine whether representation quality or scale decomposition is limiting.

## Phase 3: Heavy Research Tracks (3-6+ weeks)
- Variational unrolling
- Diffusion refinement

Deliverable:
- Confirm whether iterative methods can beat best Phase 1/2 method enough to justify ongoing cost.

## Standard Acceptance Gate (Apply to Every Method)

Advance a method only if it meets all of:
- Improves slope RMSE deg on AU full validation (`VAL_HARD_FRACTION=1.0`).
- Does not regress elevation RMSE beyond acceptable margin.
- Improves at least one hard stratum (`slope_bin` high or hydrology-sensitive bins).
- Runtime/cost stays within a practical budget envelope.

## Minimal Run Template (Per Method)

1. Pilot run:
   - Short run on fixed manifests for debugging/training stability.
2. Screening run:
   - Same budget as existing short protocol for rough ranking.
3. Full validation:
   - AU full validation set for absolute metrics.
4. Stratified analysis:
   - Use existing stratified outputs to verify where gains occur.

## Operational Notes

- Keep one canonical comparison table with:
  - elevation RMSE/MAE,
  - slope RMSE deg,
  - laplacian RMSE,
  - SDF RMSE,
  - train/inference throughput,
  - GPU-hours.
- Track non-model requirements explicitly in each experiment ticket:
  - input changes,
  - data/manifest changes,
  - eval script changes,
  - checkpoint/schema changes.
- Promote only methods with repeatable gains across at least two seeds or two AU slices.
