#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs

submit_python_job() {
  local job_name="$1"
  local dependency="${2:-}"
  shift 2

  local -a sbatch_args=(
    --parsable
    --job-name="$job_name"
    --partition=gpu
    --gres=gpu:1
    --cpus-per-task=4
    --mem=24G
    --time=12:00:00
    --output="logs/${job_name}-%j.out"
    --error="logs/${job_name}-%j.err"
  )

  if [[ -n "$dependency" ]]; then
    sbatch_args+=(--dependency="afterok:${dependency}")
  fi

  local cmd
  cmd="$(printf '%q ' "$@")"
  cmd="${cmd% }"

  sbatch "${sbatch_args[@]}" --wrap="bash -lc 'cd /home/gorelick/projects/DEM && source /home/gorelick/venv/bin/activate && ${cmd}'"
}

submit_train_eval_pair() {
  local name="$1"
  local cfg="experiment-runs/full/${name}.json"
  local train_job eval_job

  train_job="$(submit_python_job "train-${name}" "" python3 train_experiment.py --config "$cfg")"
  echo "submitted train-${name}: ${train_job}"

  eval_job="$(submit_python_job "eval-${name}" "$train_job" python3 eval_experiment.py --config "$cfg")"
  echo "submitted eval-${name}:  ${eval_job} (afterok:${train_job})"
}

for cfg in baseline multitask hydrology frequency_domain; do
  submit_train_eval_pair "$cfg"
done

ssl_cfg="experiment-runs/full/self_supervised_pretraining.json"
ssl_pretrain="$(submit_python_job "pretrain-ssl" "" python3 pretrain_experiment.py --config "$ssl_cfg")"
echo "submitted pretrain-ssl: ${ssl_pretrain}"

ssl_train="$(submit_python_job "train-ssl" "$ssl_pretrain" python3 train_experiment.py --config "$ssl_cfg")"
echo "submitted train-ssl:    ${ssl_train} (afterok:${ssl_pretrain})"

ssl_eval="$(submit_python_job "eval-ssl" "$ssl_train" python3 eval_experiment.py --config "$ssl_cfg")"
echo "submitted eval-ssl:     ${ssl_eval} (afterok:${ssl_train})"

echo
echo "All jobs submitted. Slurm will run up to two at a time on this 2-GPU node."
