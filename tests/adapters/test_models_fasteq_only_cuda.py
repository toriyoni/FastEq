"""Run complete model force training with all EQX CUDA entry points forbidden."""

import json
import os
from pathlib import Path

import pytest
import test_models_cuda as cases
import torch
from eqx_cuda_spy import forbid_eqx_cuda

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("tece", [False, True])
def test_tace_tece_fasteq_only(dtype, tece, monkeypatch):
    original = cases.compare
    recorded = []

    def compare_with_guard(reference, call, dtype, label):
        success = False

        def checked_call(model):
            if model is reference:
                return call(model)
            # Keep the context open through loss.backward: compare invokes
            # this returned mapping and then performs backward immediately.
            guard = forbid_eqx_cuda()
            recorded.append(guard)
            calls = guard.__enter__()
            recorded.append(calls)
            return call(model)

        try:
            result = original(reference, checked_call, dtype, label)
            success = True
            return result
        finally:
            guards = [x for x in recorded if not isinstance(x, list)]
            for guard in reversed(guards):
                guard.__exit__(None, None, None)
            assert all(not x for x in recorded if isinstance(x, list))
            path = Path(
                os.environ.get(
                    "FASTEQ_VALIDATION_DIR",
                    "/public-data/zhouxibo/zxb/tace_prophet_fasteq/performance/fasteq_reuse/validation",
                )
            )
            path.mkdir(parents=True, exist_ok=True)
            (
                path / (label + "_" + str(dtype).split(".")[-1] + "_eqx_spy.json")
            ).write_text(
                json.dumps(
                    {
                        "model": label,
                        "dtype": str(dtype),
                        "EQX_cuda_calls": [],
                        "complete_force_loss_backward": success,
                    },
                    indent=2,
                )
            )

    monkeypatch.setattr(cases, "compare", compare_with_guard)
    cases.test_tace_tece(dtype, tece)
