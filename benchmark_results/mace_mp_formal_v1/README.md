# MACE-MP-0 force-training operator benchmark

This directory preserves the FP32 e3nn versus FastEq Uniform1D benchmark for
the official MACE-MP-0 `small`, `medium`, and `large` checkpoints.

## Scope

The timed region is one real force-training compute step:

1. MACE forward with `training=True` and `compute_force=True`;
2. force loss `mean(forces ** 2)`;
3. `loss.backward()`, which traverses the mixed second-derivative path needed
   for force training.

Model loading, graph construction, host/device copies, `zero_grad`, optimizer
updates, data loading, checkpoint I/O, and epoch orchestration are excluded.
This is therefore a training-operator/end-to-end model-compute benchmark, not
a complete wall-clock fine-tuning run.

## Provenance

- OneScience source repository: `https://github.com/hpccube/OneScience`
- Inspected OneScience commit: `4aa6964bcfe8c3aa72d119574d76357f2890ae51`
- OneScience entry point: `examples/matchem/mace/train.py`
- Model selection used by that entry point: `--foundation_model small|medium|large`
- FastEq commit under test: `ecaa96fbb011cf4a2ab63b6dfd63025148bf7ce5`
- MACE submodule commit: `1a3feb9d17eb41ef40739364d8ef3034da00acf3`
- Host/GPU: `gxn70`, NVIDIA H100 80GB HBM3, GPU 7
- Python/PyTorch: Python 3.10.20, PyTorch 2.11.0+cu128
- Environment: `/public-data/zhouxibo/miniconda3/envs/fasteq_v0.3.0`

`fasteq_uniform1d` converts only Uniform1D tensor products to the new FastEq
path; other convertible operations keep the cuEquivariance implementation.
All FastEq Uniform1D tuning switches were disabled for this reproducible first
baseline, so the generated candidate policy is `w1only`.

## Protocol

- dtype: FP32 (official FP64 checkpoints converted by MACE at load time)
- structures: replicated 3-atom water cell with box sides 2, 4, and 6
- atom counts: 24, 192, and 648
- warmups: 20
- timed repetitions: 50
- reported timing: median CUDA-event time
- memory: peak CUDA allocation reported by the driver
- correctness: side 2, e3nn reference, `atol=1e-4`, `rtol=1e-3`

Each `side*_*/` directory contains the exact `metadata.json`, all 50 samples in
`raw.csv`, and the one-row `summary.csv`. `summary_all.csv` concatenates the 18
runs. `comparison.csv` contains speedup, memory reduction, and correctness
errors. The adjacent `.log` files retain the final console record.

The model binaries are intentionally not committed. Download them using
`benchmark_results/models/MODELS.md`, verify SHA256, then run
`benchmark_results/mace_mp_formal_v1/run_mace_mp_training_sweep.sh`.
