#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

for cfg in baseline multitask hydrology frequency_domain; do
  config_path="experiment-runs/full/${cfg}.json"
  python3 train_experiment.py --config "${config_path}"
  python3 eval_experiment.py --config "${config_path}"
done

ssl_config="experiment-runs/full/self_supervised_pretraining.json"
python3 pretrain_experiment.py --config "${ssl_config}"
python3 train_experiment.py --config "${ssl_config}"
python3 eval_experiment.py --config "${ssl_config}"
