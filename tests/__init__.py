"""Project-wide unittest package (imports as tests.*).

Layout mirrors domains:
  core/          — checkpoints, metrics, schema, config, run_config, reporting
  experiments/ — registry, named presets (cross-plan)
  train/         — train.engine, train_experiment resume
  eval/          — eval.engine, sliding-window inference
  losses/        — composite / component losses
  models/        — stack layout, model factory, eval stratify constants
  pretraining/   — encoder SSL helpers
  ingest/        — TDEM/EDEM ingest pipeline
  entrypoints/   — CLI parsing for train/eval/pretrain scripts

Per-experiment model tests live under experiments/<name>/test_*.py alongside plans.
"""
