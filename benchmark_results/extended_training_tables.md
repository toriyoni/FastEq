# MACE force-training extended sweep: e3nn vs FastEq Uniform1D

Protocol: MACE-OFF23 Small, FP32, NVIDIA H100 80GB, 20 warmups, 50 timed repeats, CUDA-event medians. The timed step is `training=True` force forward plus `mean(force^2).backward()`; graph construction, transfers, `zero_grad`, and optimizer update are excluded.

## Table 1. End-to-end force-training performance

| Atoms | Edges | e3nn FWD ms | e3nn BWD ms | e3nn total ms | FastEq FWD ms | FastEq BWD ms | FastEq total ms | e3nn/FastEq | FastEq time saved | e3nn GiB | FastEq GiB | Memory saved | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 192 | 7872 | 13.254 | 13.380 | 26.886 | 17.985 | 14.190 | 32.203 | 0.83x | -19.8% | 0.543 | 0.251 | 53.8% | 双方通过 |
| 648 | 26526 | 16.463 | 21.446 | 37.792 | 18.316 | 14.327 | 32.920 | 1.15x | 12.9% | 1.647 | 0.681 | 58.6% | 双方通过 |
| 1536 | 63020 | 19.089 | 35.288 | 54.432 | 18.203 | 14.916 | 33.194 | 1.64x | 39.0% | 3.814 | 1.528 | 59.9% | 双方通过 |
| 3000 | 122788 | 34.374 | 63.740 | 98.116 | 19.138 | 24.340 | 43.494 | 2.26x | 55.7% | 7.346 | 2.872 | 60.9% | 双方通过 |
| 5184 | 212318 | 57.532 | 107.109 | 164.655 | 29.543 | 40.274 | 69.854 | 2.36x | 57.6% | 12.648 | 4.915 | 61.1% | 双方通过 |
| 12288 | 503396 | 130.827 | 248.246 | 379.084 | 57.158 | 91.002 | 148.173 | 2.56x | 60.9% | 29.881 | 11.541 | 61.4% | 双方通过 |
| 24000 | 983516 | 251.779 | 481.465 | 733.264 | 99.149 | 173.871 | 273.094 | 2.69x | 62.8% | 58.313 | 22.476 | 61.5% | 双方通过 |
| 41472 | 1698590 | — | — | — | 156.995 | 295.708 | 452.744 | — | — | — | 38.770 | — | e3nn 实测 OOM；FastEq 通过 |
| 65856 | 2699118 | — | — | — | 240.978 | 467.404 | 708.418 | — | — | — | 61.541 | — | e3nn 超实测 OOM 边界；FastEq 通过 |
| 98304 | — | — | — | — | — | — | — | — | — | — | — | — | FastEq 实测 OOM；e3nn 已超边界 |
| 139968 | — | — | — | — | — | — | — | — | — | — | — | — | 双方均超实测 OOM 边界，未重复运行 |
| 192000 | — | — | — | — | — | — | — | — | — | — | — | — | 双方均超实测 OOM 边界，未重复运行 |
| 222264 | — | — | — | — | — | — | — | — | — | — | — | — | 双方均超实测 OOM 边界，未重复运行 |
| 255552 | — | — | — | — | — | — | — | — | — | — | — | — | 双方均超实测 OOM 边界，未重复运行 |

## Table 2. FastEq numerical and force-loss gradient error vs e3nn

| Atoms | Edges | Energy abs | Energy rel | Forces abs | Forces rel | Loss abs | Loss rel | Worst comparable gradient | Grad abs | Grad rel | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|---:|---:|:---|
| 192 | 7872 | 1.562e-02 | 1.173e-07 | 2.176e-06 | 1.648e-06 | 2.980e-08 | 1.638e-07 | node_embedding.linear.weight | 6.706e-08 | 5.443e-07 | 通过 |
| 648 | 26526 | 6.250e-02 | 1.390e-07 | 3.368e-06 | 1.485e-06 | 2.980e-08 | 1.471e-07 | node_embedding.linear.weight | 1.155e-07 | 9.630e-07 | 通过 |
| 1536 | 63020 | 0.000e+00 | 0.000e+00 | 4.034e-06 | 2.041e-06 | 1.490e-08 | 6.833e-08 | interactions.0.conv_tp_weights.layer0.weight | 7.451e-08 | 3.363e-07 | 通过 |
| 3000 | 122788 | 0.000e+00 | 0.000e+00 | 3.606e-06 | 1.708e-06 | 0.000e+00 | 0.000e+00 | node_embedding.linear.weight | 6.333e-08 | 4.930e-07 | 通过 |
| 5184 | 212318 | 2.500e-01 | 6.952e-08 | 4.053e-06 | 1.916e-06 | 0.000e+00 | 0.000e+00 | node_embedding.linear.weight | 3.353e-08 | 2.630e-07 | 通过 |
| 12288 | 503396 | 1.000e+00 | 1.173e-07 | 4.798e-06 | 2.176e-06 | 2.980e-08 | 1.404e-07 | node_embedding.linear.weight | 5.588e-08 | 4.424e-07 | 通过 |
| 24000 | 983516 | 1.000e+00 | 6.005e-08 | 5.871e-06 | 2.515e-06 | 2.980e-08 | 1.413e-07 | interactions.0.conv_tp_weights.layer0.weight | 5.960e-08 | 2.749e-07 | 通过 |
| 41472 | 1698590 | — | — | — | — | — | — | — | — | — | 无 e3nn 参考：e3nn OOM |
| 65856 | 2699118 | — | — | — | — | — | — | — | — | — | 无 e3nn 参考：e3nn OOM |
| 98304 | — | — | — | — | — | — | — | — | — | — | 双方超显存边界 |
| 139968 | — | — | — | — | — | — | — | — | — | — | 双方超显存边界 |
| 192000 | — | — | — | — | — | — | — | — | — | — | 双方超显存边界 |
| 222264 | — | — | — | — | — | — | — | — | — | — | 双方超显存边界 |
| 255552 | — | — | — | — | — | — | — | — | — | — | 双方超显存边界 |

Notes:

- e3nn first failed at 41,472 atoms; FastEq first failed at 98,304 atoms.
- Larger rows were not repeatedly executed after the monotonic OOM boundary was established.
- Accuracy uses `atol=5e-5`, `rtol=5e-4`, compares 20 directly name-aligned parameter tensors, and explicitly covers all eight `conv_tp_weights` tensors. Symmetric-contraction parameters are packed differently between e3nn and cuEquivariance and are reported separately in the source JSON files.
