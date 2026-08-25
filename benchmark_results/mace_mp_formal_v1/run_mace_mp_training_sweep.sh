#!/usr/bin/env bash
set -uo pipefail

repo_root=${1:-/public-data/zhouxibo/zxb/FastEq}
env_root=${2:-/public-data/zhouxibo/miniconda3/envs/fasteq_v0.3.0}
gpu_index=${CUDA_VISIBLE_DEVICES:-7}

source "${env_root}/bin/activate"
cd "${repo_root}"

export CUDA_VISIBLE_DEVICES="${gpu_index}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export FASTEQ_UNIFORM1D_FWD_TUNE=0
export FASTEQ_UNIFORM1D_BWD_TUNE=0
export FASTEQ_UNIFORM1D_DOUBLE_BWD_TUNE=0

result_root=benchmark_results/mace_mp_formal_v1
mkdir -p "${result_root}"

for side in 2 4 6; do
  for size in small medium large; do
    for backend in e3nn fasteq_uniform1d; do
      output_dir="${result_root}/side${side}_${size}_${backend}"
      python test/mace_training_bench.py \
        --backend "${backend}" \
        --model "benchmark_results/models/MACE-MP-0_${size}.model" \
        --dtype float32 \
        --device cuda \
        --box-sides "${side}" \
        --warmup 20 \
        --repeats 50 \
        --output-dir "${output_dir}" \
        > "${output_dir}.log" 2>&1
      sed -i 's/\r$//' "${output_dir}/raw.csv" "${output_dir}/summary.csv"
      tail -n 1 "${output_dir}.log"
    done
  done
done

python benchmark_results/mace_mp_formal_v1/aggregate_mace_mp_results.py \
  --root "${result_root}" \
  --correctness-root benchmark_results/mace_mp_correctness
