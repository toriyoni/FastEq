# Official MACE-MP-0 checkpoints

The OneScience MACE training entry point resolves `--foundation_model` through
its `mace_mp()` helper. This benchmark uses the same three official checkpoint
URLs.

| Size | URL | SHA256 |
| --- | --- | --- |
| small | `https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model` | `2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736` |
| medium | `https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model` | `01bfe22100139f424713cf921144e5509cbe353d67aa9fa1be9c6e1e0ed35845` |
| large | `https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/MACE_MPtrj_2022.9.model` | `f80e992b65ab8f88fdf26964511357c022e92704e4d9bcd086652635a8495b32` |

Expected local names:

- `benchmark_results/models/MACE-MP-0_small.model`
- `benchmark_results/models/MACE-MP-0_medium.model`
- `benchmark_results/models/MACE-MP-0_large.model`

The checkpoint files are excluded from Git because they are approximately
32 MB, 43 MB, and 128 MB respectively.
