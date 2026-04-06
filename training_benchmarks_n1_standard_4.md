# Training Benchmarks: `n1-standard-4`

Benchmarks collected on the upgraded GCE VM for comparison with the earlier 2-vCPU T4 host.

## Machine

- GPU: `1x NVIDIA T4`
- CPU: `4 vCPU`
- RAM: `~15 GiB`
- Runtime env: `/home/gorelick/venv-cu128`
- Python: `3.12`
- PyTorch: `2.7.1+cu128`
- Rasterio: `1.5.0`
- Data root: `/data/training`
- Train manifest: `train_manifest_seed42.txt`
- Train patches: `135469`

## Benchmark Setup

- `batch_size=16`
- AMP enabled
- Same `DemFilmUNet` model and `loss_dem`
- Same local GeoTIFF loading and non-finite sanitization logic as the previous host
- End-to-end benchmark uses `100` batches
- Compute-only benchmark uses repeated steps on one already-loaded batch

## End-to-End Training Throughput

| workers | sec_per_batch | patches_per_sec |
| --- | ---: | ---: |
| 0 | 0.7378 | 21.69 |
| 1 | 0.5190 | 30.83 |
| 2 | 0.3100 | 51.61 |
| 3 | 0.2881 | 55.53 |
| 4 | 0.2874 | 55.67 |

Takeaway: throughput improves strongly up to `workers=3`, then plateaus.

## Compute-Only Training Throughput

| mode | sec_per_batch | patches_per_sec |
| --- | ---: | ---: |
| compute_only | 0.1173 | 136.36 |

Takeaway: the upgraded host gets much closer to the GPU-only floor than the previous 2-vCPU host.

## Comparison To Previous 2-vCPU Host

Earlier benchmark reference at `batch_size=16`:

| host | workers | sec_per_batch | patches_per_sec |
| --- | ---: | ---: | ---: |
| previous T4 host | 1 | 0.6775 | 23.62 |
| previous T4 host | 2 | 0.5879 | 27.22 |
| `n1-standard-4` | 1 | 0.5190 | 30.83 |
| `n1-standard-4` | 2 | 0.3100 | 51.61 |
| `n1-standard-4` | 3 | 0.2881 | 55.53 |
| `n1-standard-4` | 4 | 0.2874 | 55.67 |

Speedups:

- New host `workers=1` vs old host `workers=1`: `1.31x`
- New host `workers=2` vs old host `workers=2`: `1.90x`
- New host best (`workers=4`) vs old host best (`workers=2`): `2.05x`

## Estimated Full-Run Time

Using `135469` training patches and `batch_size=16`:

- Steps per epoch: `ceil(135469 / 16) = 8467`
- Best measured time per batch: `0.2874 s`
- Estimated time per epoch: `40.5 min`
- Estimated time for 3 epochs: `2.03 hr`

## Recommended Setting On This Host

- `batch_size=16`
- `workers=3`
- `--amp`

Reasoning:

- `workers=4` is only marginally faster than `workers=3`
- `workers=3` should leave a little headroom for OS/background overhead
