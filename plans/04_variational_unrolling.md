# Plan: Physics-Inspired Variational Unrolling

## Goal
Unroll an optimization process with learnable proximal steps instead of direct black-box regression.

## Why This Is Different
- Current approach is pure supervised mapping.
- Unrolling combines data fidelity and explicit terrain priors in iterative updates.

## Implementation Plan
1. Define objective terms:
  - fidelity to observed/baseline structure,
  - smoothness/anisotropic regularization,
  - contour-aware prior term.
2. Implement `K` unrolled iterations with learnable step sizes and proximal blocks.
3. Train end-to-end on weighted pixel loss plus objective consistency terms.
4. Tune iteration count for speed/quality tradeoff.

## Required Beyond New Model
- **Objective-definition work**
  - Formalize terrain priors and differentiable penalties.
  - Decide fixed vs learnable regularization weights.
- **Training pipeline changes**
  - Add stability controls (gradient clipping, step-size constraints).
  - Add logging of per-iteration objective values.
- **Inference updates**
  - Expose iteration count and optional early-stop criteria.
- **Evaluation additions**
  - Track physical plausibility diagnostics (e.g., roughness spikes, curvature outliers).

## Data and Preprocessing Needs
- Existing inputs can be used directly.
- May need additional quality masks for robust fidelity weighting in noisy regions.

## Evaluation Plan
- Compare standard metrics and plausibility diagnostics.
- Validate generalization to AU strata with weak supervision quality.

## Risks
- Harder optimization and sensitivity to hyperparameters.
- Can be slower than feed-forward models at inference.

## Pilot Exit Criteria
- Keep if it improves robustness on hard strata and reduces curvature artifacts.
