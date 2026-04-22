# Plan 07 Self-Supervised Pretraining (Full/Full)

This run directory defines a two-phase workflow:

- phase 1: masked-reconstruction SSL pretraining for DEM+AE encoders
- phase 2: supervised finetuning with unchanged downstream objective

## 1) Pretrain encoder

```bash
python3 pretrain_experiment.py --config experiment-runs/07_self_supervised_pretraining_full_full/run_config.json
```

Outputs:

- `experiment-runs/07_self_supervised_pretraining_full_full/checkpoints/ssl_encoder_pretrain.pt`
- `experiment-runs/07_self_supervised_pretraining_full_full/pretrain_report.json`

## 2) Finetune supervised model

```bash
python3 train_experiment.py --config experiment-runs/07_self_supervised_pretraining_full_full/run_config.json
```

This automatically loads:

- `train.pretrained_encoder_checkpoint`

into the supervised model encoder before training.

## 3) Evaluate finetuned model

```bash
python3 eval_experiment.py --config experiment-runs/07_self_supervised_pretraining_full_full/run_config.json
```

Since `eval.output_json` is `null`, standardized eval output is written under:

- `experiment-runs/07_self_supervised_pretraining_full_full/results/`
