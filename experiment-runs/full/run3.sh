#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

python3 eval_experiment.py --config experiment-runs/full/baseline.json
python3 eval_experiment.py --config experiment-runs/full/multitask.json
python3 eval_experiment.py --config experiment-runs/full/hydrology.json
python3 eval_experiment.py --config experiment-runs/full/frequency_domain.json
python3 eval_experiment.py --config experiment-runs/full/self_supervised_pretraining.json
