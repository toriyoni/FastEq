# Equivariant LayerNorm

FastEq fuses equivariant normalization for EquiformerV3 and EquiformerV2 layers,
with GPU FP32 forward execution and first-order gradients on CUDA and HIP.

MergeLayerNorm supports the normalization used by native EQv3 inference.
Current forward comparisons pass on both devices. One large Hygon Merge
parameter-gradient case exceeds tolerance and remains a known limitation;
the regression pass does not imply acceptance at every shape.

## Supported layers

Inputs have shape `[N, (L + 1)**2, C]`, where `N` is the number of atoms, `L` is
the maximum degree, and `C` is the channel count. Statistics are computed
independently for each atom. Let `q_l` be the mean squared feature value across
the `2*l + 1` components and `C` channels of degree `l`, after scalar centering.

| Model | Native layer | Default normalization statistics |
| --- | --- | --- |
| EquiformerV3 | `EquivariantLayerNorm` | Each degree uses its own `q_l`. |
| EquiformerV3 | `EquivariantMergeLayerNorm` | All degrees share `mean(q_0, ..., q_L)`. |
| EquiformerV3 | `EquivariantSeparableLayerNorm` | Scalars use `q_0`; higher degrees share `mean(q_1, ..., q_L)`. |
| EquiformerV2 | `EquivariantLayerNormArraySphericalHarmonics` | Scalars use `q_0`; higher degrees share `mean(q_1, ..., q_L)`. |

The table uses the native defaults, including equal weighting of degrees where
applicable. The adapter preserves the source layer's normalization options.
Only scalar features (`l = 0`) are centered across channels. Affine scales are learned per degree and channel and shared across
that degree's components. Bias applies only to scalars when the source has it.

EQv3 factory names `merge_layer_norm` and `merge_layer_norm_attn_rms_norm`
create `EquivariantMergeLayerNorm`; `merge_rms_norm` uses the same class with
scalar centering disabled. The adapter preserves this choice. It does not
replace the separate ordinary RMSNorm used inside attention. `N` is the total
atom count in one layer call, including all structures in a batch.

## Fused operations

Forward combines scalar mean, centering, squaring, weighted group reduction,
reciprocal square root with epsilon, and affine transformation in one kernel.
Each group supplies its normalization scale to the degrees listed above.

For native source adapters, Torch computes the source-ordered statistic-gradient
reductions before a shared Triton program produces `dX` and per-atom affine
parameter contributions. Torch then sums contributions across atoms, and a
shared Triton kernel writes the degree-shared parameter gradients. The same
code and reduction APIs run on CUDA and HIP; no backend-specific module is
selected. This splits part of backward fusion to preserve native FP32 tolerance
without reimplementing each backend's reduction scheduler.

Group boundaries are device buffers, avoiding unsupported `constexpr` tuple
indexing. Hardware lane width is read from the active Triton target; a
32-channel tile is independent of warp/wavefront width. The implementation uses
ordinary Triton expressions without CUDA libdevice calls or inline assembly.
The supported presets are `per_degree` (the default), `scalar_high`, and `all`
(MergeLayerNorm). All three use the same CUDA/HIP computational kernels.

Only requested gradients are computed. Double backward and `create_graph=True`
are unsupported. Source adapters default to `parameter_reduction="native"`.
The optional `parameter_reduction="stable"` policy uses fixed 256-row Triton
reduction trees, with `accumulation="fp32"` or `"fp64"` controlling cross-atom
accumulators. These are explicit numerical policies shared by both backends.
FP64 is a diagnostic reference, not a replacement for Torch FP32 acceptance.

## Usage

```python
from fasteq.triton import from_reference

# source is the original FP32 GPU layer; x has shape [N, (L + 1)**2, C].
op = from_reference(source)
y = op(x.requires_grad_(True))
y.square().mean().backward()
```

For a native MergeLayerNorm in a fixed-weight inference model:

```python
import torch
from fasteq.triton import from_reference

model.norm = from_reference(model.norm).eval()
with torch.inference_mode():
    y = model.norm(x)
```

The adapter reuses the source parameters, so gradients reach those parameters.
Apply it to the individual normalization layers you want to replace. Both
contiguous NKC storage and KNC storage viewed as `[N, K, C]` are supported,
where `K = (L + 1)**2`. CUDA and HIP FP32 are supported execution targets.

## Current-source validation

The 2026-09-20 runs use the same operator and test sources on both platforms.
The runtime SHA-256 is
`c1b2f19f8287ed352ab5a0579469eb797a0ef8b83276b923eaf3d1408b7e0d8b`.
The [result summary](validation/layernorm/summary.json) pins all eight runtime,
reference-helper, test and benchmark files.

| Platform | Host / device | Torch | Triton | LayerNorm regression | Merge size matrix | Captured-input replay |
| --- | --- | --- | --- | --- | --- | --- |
| H100 | gxn70, physical GPU 0, H100 80GB HBM3, sm90 | 2.11.0+cu128 | 3.6.0 | 372 passed, zero failures/errors/skips | 40/40 passed | 12/12 passed |
| Hygon | f11r1n05, one BW DCU, gfx936 | 2.7.1 / HIP 6.3.26045 | HCU 3.1.0 | 372 passed, zero failures/errors/skips | 39/40 passed | 12/12 passed |

The regression suite covers per-degree, Merge, Separable and V2 spherical
harmonics normalization: output, `dX`, affine gradients, NKC/KNC layouts,
irregular channels, source factory options, scalar centering, affine on/off,
component/norm weighting, parameter sharing, cancellation and SGD steps.
Comparisons use unmodified native Torch FP32 layers with ordinary elementwise
`atol=5e-5, rtol=5e-4`; stricter mathematical tests are retained. No tolerance
was relaxed and no FP64 result replaced Torch acceptance.

References:

- EQv3 `layer_norm.py` at `a7300c58df683dc99cb48027d5bfd4c887486c48`, SHA-256 `41e00b377cbacef4713d88697c06e419f3b8c208d9fcc55be78b53fd5a23c2c6`.
- EQv2 `layer_norm.py` at `d5ad4be729b56f74012ebb7f097f77c5b00a1004`, SHA-256 `fdd7955f1fc3ef4e0a0ccbfb761128b5b269ad09f50423428f77aee12a53480c`.

The Merge size matrix fixes C=128, uses Lmax=3 and 6, and doubles N from 256
through 131,072. Each of the ten sizes is checked in forward and forward plus
backward mode, giving 40 cases per device. **This is a bounded operator test,
not an OOM sweep or a whole-model capacity measurement.** All 20 forward
cases pass on each device; all sampled sizes up to N=65,536 pass both modes.
The current source has no new standalone per-degree, Separable or V2
performance sweep; older implementation timings are omitted. Restoring the
Merge adapter does not change the reduction kernels or establish that earlier
Separable cancellation boundaries have been fixed.

### Known numerical limitation

On Hygon, `N=131072, Lmax=6, C=128`, one of 896 `affine_weight` gradient
entries exceeds tolerance, at `[degree=2, channel=125]`:

| FastEq | Native Torch FP32 | Absolute error | Allowed error | Error / tolerance |
| ---: | ---: | ---: | ---: | ---: |
| 0.125518798828125 | 0.1253662109375 | 0.000152587890625 | 0.00011268310546875 | 1.354133 |

This is approximately 0.122% relative error in that parameter-gradient entry,
not an energy or force error. Forward, `dX` and bias gradients pass at the same
shape; the H100 case passes. The failure reproduced three times on Hygon node
f11r1n01 using the same sources and seed. It is retained as an accepted known
limitation of this release, not marked as a passed numerical comparison.

Fixed-weight forward inference does not use parameter gradients. The tested
forward path therefore has no failure associated with this boundary; full-model
energy/force accuracy, long trajectories, force-loss training and higher-order
autograd are not established by these layer tests. Cancellation also depends
on input values, so sampled passing sizes are not universal safety thresholds.

## MergeLayerNorm performance

All timings are FP32 standalone layer calls, with five warmups and 20 samples
alternating native Torch and FastEq order. Tables use GPU-event medians;
synchronized wall times and all samples are retained. Forward uses `eval()`
and `torch.inference_mode()`. Forward + backward includes `dX`, all affine
parameter gradients and the Torch reductions in the adapter, without an
optimizer. The H100 host is shared and clocks were not locked; speedups compare
implementations on the same device, not hardware throughput across platforms.

### Captured inference inputs

Six inputs and EMA parameter states come from native `equiformer_v3_dens`
inference: a 108-atom Cu structure and a 280-atom mixed batch, each at
`blocks.0.norm_1`, `blocks.0.norm_2` and final `norm`. Shapes are `[N,25,128]`
(Lmax=4), including noncontiguous inputs. Current native Torch and FastEq
outputs are compared with the saved outputs. First-order backward additionally
uses the same random upstream gradient for both implementations.

| Device | Forward speedup | Forward + backward speedup | Maximum error / tolerance across replay checks |
| --- | ---: | ---: | ---: |
| H100 | 1.64–1.75x | 1.23–1.33x | 0.015170 |
| Hygon BW | 1.84–1.89x | 1.38–1.41x | 0.037926 |

These are 12 passing standalone replays per device, not a new end-to-end model
run or a whole-model speedup. Capture provenance includes source commit,
checkpoint SHA-256, EMA weight selection and fixture hashes in the raw replay
files. The captured checkpoint has 7 layers, Lmax=4 and 128 channels; the random
matrix also covers Lmax=6.

### Representative 8,192-atom cases

All entries below pass the complete forward and requested gradient checks.

| Device | Lmax | Mode | Torch ms | FastEq ms | Speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| H100 | 3 | Forward | 0.3549 | 0.1469 | 2.42x |
| H100 | 3 | Forward + backward | 1.4154 | 0.7414 | 1.91x |
| H100 | 6 | Forward | 0.8967 | 0.5393 | 1.66x |
| H100 | 6 | Forward + backward | 3.6716 | 1.6980 | 2.16x |
| Hygon BW | 3 | Forward | 1.1396 | 0.4273 | 2.67x |
| Hygon BW | 3 | Forward + backward | 4.1033 | 1.5423 | 2.66x |
| Hygon BW | 6 | Forward | 3.0713 | 1.6081 | 1.91x |
| Hygon BW | 6 | Forward + backward | 11.8035 | 4.5518 | 2.59x |

### Bounded size matrix

Only accuracy-passing points contribute to the ranges. The failed Hygon
backward case has no accepted timing.

| Device | Lmax | Forward cases | Forward speedup | Forward + backward cases | Forward + backward speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| H100 | 3 | 10/10 | 1.48–3.99x | 10/10 | 1.27–2.75x |
| H100 | 6 | 10/10 | 1.20–1.83x | 10/10 | 1.26–2.30x |
| Hygon BW | 3 | 10/10 | 1.52–4.71x | 10/10 | 1.38–3.33x |
| Hygon BW | 6 | 10/10 | 1.36–2.12x | 9/10 | 1.40–2.75x |

## Records and reproduction

- [Current summary, source hashes and all timing/precision metrics](validation/layernorm/summary.json)
- [H100 JUnit](validation/layernorm/h100/pytest.xml) and [Hygon JUnit](validation/layernorm/hygon/pytest.xml)
- [H100 raw size matrix](validation/layernorm/h100/matrix.json.gz) and [Hygon raw size matrix](validation/layernorm/hygon/matrix.json.gz)
- [H100 captured-input replay](validation/layernorm/h100/replay.json) and [Hygon captured-input replay](validation/layernorm/hygon/replay.json)
- [Known Hygon boundary](validation/layernorm/hygon/known_boundary.json) and [original failing comparison](validation/layernorm/hygon/merge_l6_fwd_bwd_131072.log)

Matrix archives map the original JSON basename to the unchanged result;
`raw_record` in the summary selects that entry. The [source manifest](validation/manifest.json)
and [reproduction commands](validation/REPRODUCE.md) cover the four regression
modules, bounded benchmark and fixture replay. The GPU-free record checker
verifies sources, metrics, timing samples and the retained failure.
