#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs experiment-runs/sweeps/bs-lr-workers

# Bounded sweep so results return quickly on shared hardware.
batch_sizes=(32 64 128)
learning_rates=(0.0001 0.0002 0.0004)
workers_list=(4 8)

manifest="patches/training_manifest.txt"
data_root="./data/training"
epochs=2
max_patches=8192
experiment="baseline"
arch="film_unet"
seed=42

submit_train_job() {
  local bs="$1"
  local lr="$2"
  local workers="$3"
  local run_id="bs${bs}_lr${lr}_w${workers}"
  run_id="${run_id//./p}"
  local out_dir="experiment-runs/sweeps/bs-lr-workers/${run_id}"
  local ckpt="${out_dir}/checkpoints/model.pt"
  local report="${out_dir}/train_report.json"
  local log_prefix="sweep-${run_id}"

  mkdir -p "${out_dir}/checkpoints"

  local cpus="${workers}"
  if [[ "${cpus}" -lt 4 ]]; then
    cpus=4
  fi

  local cmd
  cmd="$(cat <<EOF
python3 train_experiment.py \
  --experiment ${experiment} \
  --arch ${arch} \
  --data-root ${data_root} \
  --manifest ${manifest} \
  --epochs ${epochs} \
  --max-patches ${max_patches} \
  --batch-size ${bs} \
  --lr ${lr} \
  --workers ${workers} \
  --seed ${seed} \
  --amp \
  --checkpoint-out ${ckpt} \
  --output-json ${report}
EOF
)"

  local job_id
  job_id="$(
    sbatch --parsable \
      --job-name="${log_prefix}" \
      --partition=gpu \
      --gres=gpu:1 \
      --cpus-per-task="${cpus}" \
      --mem=24G \
      --time=06:00:00 \
      --output="logs/${log_prefix}-%j.out" \
      --error="logs/${log_prefix}-%j.err" \
      --wrap="bash -lc 'cd /home/gorelick/projects/DEM && source /home/gorelick/venv/bin/activate && ${cmd}'"
  )"

  echo "${run_id},${job_id}"
}

echo "run_id,job_id" > experiment-runs/sweeps/bs-lr-workers/submitted_jobs.csv
for bs in "${batch_sizes[@]}"; do
  for lr in "${learning_rates[@]}"; do
    for workers in "${workers_list[@]}"; do
      result="$(submit_train_job "${bs}" "${lr}" "${workers}")"
      echo "${result}" | tee -a experiment-runs/sweeps/bs-lr-workers/submitted_jobs.csv
    done
  done
done

echo
echo "Submitted sweep jobs:"
cat experiment-runs/sweeps/bs-lr-workers/submitted_jobs.csv
