"""CPU checks for new gradient-chain and capacity classification machinery."""

import unittest

import torch
from benchmark_scaling_case import (
    SharedGPUError,
    classify_failure,
    compact_snapshot,
    compare_snapshots,
    execute_operator,
    logical_sample,
    verify_finite,
)


class ScalingWorkerTests(unittest.TestCase):
    def test_only_cuda_allocation_failure_is_boundary(self):
        self.assertEqual(classify_failure(RuntimeError("CUDA out of memory"), "warmup"),
                         ("oom", "cuda_oom", True))
        self.assertEqual(classify_failure(RuntimeError("CUDA out of memory"), "validation"),
                         ("error", "validation", False))
        self.assertEqual(classify_failure(MemoryError("allocation"), "setup"),
                         ("error", "cpu_oom", False))
        self.assertEqual(classify_failure(RuntimeError("CUDA illegal memory access"), "warmup"),
                         ("error", "other", False))
        self.assertEqual(classify_failure(SharedGPUError("foreign"), "warmup"),
                         ("shared_gpu", "foreign_compute_process", False))

    def test_noncontiguous_sampling_and_finite_checks(self):
        view = torch.arange(6400).reshape(100, 64)[:, 11:21]
        self.assertFalse(view.is_contiguous())
        expected = view.reshape(-1)[torch.linspace(0, view.numel() - 1, 64).long()].double()
        torch.testing.assert_close(logical_sample(view, torch, 64), expected)
        result = verify_finite({"view": view}, [], torch)
        self.assertEqual(result["elements_checked"], view.numel())
        corrupt = view.double()
        corrupt[-1, -1] = float("nan")
        with self.assertRaises(AssertionError):
            verify_finite({"view": corrupt}, [], torch)

    def test_operator_parameter_gradients_are_not_sampled(self):
        weight = torch.nn.Parameter(torch.randn(128, 8, dtype=torch.float64))
        unused = torch.nn.Parameter(torch.randn(3, dtype=torch.float64))
        value = torch.randn(3, 8, dtype=torch.float64).requires_grad_()
        case = {"inputs": (value,), "baseline_parameters": (weight, unused),
                "baseline": lambda x: x @ weight.T}
        for mode in ("backward", "force_loss"):
            output = execute_operator(case, "original", mode, torch)
            snapshot = compact_snapshot(output, [], torch)
            gradient = snapshot["output/parameter_gradients/0"]["values"]
            self.assertEqual(gradient.numel(), weight.numel())
            self.assertIsNone(snapshot["output/parameter_gradients/1"])
            compare_snapshots(snapshot, snapshot, "float64", torch)
            damaged = {key: entry for key, entry in snapshot.items()}
            damaged["output/parameter_gradients/0"] = {"shape": list(weight.shape),
                                                        "values": gradient.clone()}
            damaged["output/parameter_gradients/0"]["values"][-1, -1] += 1
            with self.assertRaises(AssertionError):
                compare_snapshots(snapshot, damaged, "float64", torch)

    def test_tuple_views_support_higher_order_input_chain(self):
        value = torch.randn(8, 32, dtype=torch.float64).requires_grad_()
        case = {"inputs": (value,), "baseline_parameters": (),
                "baseline": lambda x: (x[:, :3], x[:, 20:24])}
        output = execute_operator(case, "original", "force_loss", torch)
        self.assertEqual(output["gradients"][0].shape, value.shape)
        self.assertTrue(torch.isfinite(output["gradients"][0]).all())
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
