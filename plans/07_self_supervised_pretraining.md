# Plan: Self-Supervised Pretraining + Supervised Finetune

## Goal
Pretrain encoders on unlabeled DEM/AE data, then finetune on supervised DEM correction.

## Why This Is Different
- Current training is fully supervised from random initialization.
- Pretraining can improve representations and cross-region transfer.

## Implementation Plan
1. Select SSL objective (masked reconstruction, contrastive pairs, or hybrid).
2. Pretrain DEM and AE encoders jointly or separately on large unlabeled corpus.
3. Transfer encoder weights to supervised model and finetune with current losses.
4. Compare frozen-encoder, partial-unfreeze, and full finetune settings.

## Required Beyond New Model
- **Input/data changes**
  - Build unlabeled pretraining manifest(s), potentially much larger than current supervised set.
  - Define augmentations appropriate for topography (avoid harmful transforms).
- **Training pipeline changes**
  - New pretraining script and checkpoint format.
  - Weight-loading bridge from SSL checkpoints into `train_experiment.py`.
- **Experiment tracking changes**
  - Track pretrain dataset, objective, epochs, and transfer recipe as required metadata.
- **Compute planning**
  - Additional pretraining phase budget before supervised runs.

## Data and Preprocessing Needs
- Access to broad unlabeled DEM/AE chips (can include non-AU and AU).
- Need deduplication / leakage checks against validation patches.

## Evaluation Plan
- Compare same supervised architecture with and without SSL init.
- Focus on AU generalization and hard-strata gains.

## Risks
- SSL objective mismatch with downstream task.
- Added complexity may not pay off if dataset is already sufficient.

## Pilot Exit Criteria
- Keep if SSL init yields consistent AU gains across at least two supervised architectures.
