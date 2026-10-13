"""CPU checks for new scaling inputs, without rerunning adapter CUDA QA."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks/adapters"
sys.path.insert(0, str(BENCHMARKS))
from scaling_geometry import _template, build_scaling_data


@pytest.mark.parametrize("atoms", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_preserves_original_small_geometry(monkeypatch, atoms, dtype):
    # Execute the actual original builder with its CUDA copies redirected to
    # CPU. This checks all old input keys/values, rather than mirroring it.
    spec = importlib.util.spec_from_file_location("_scaling_original_geometry", BENCHMARKS / "benchmark_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda value, *args, **kwargs: value)
    # The old builder passes device='cuda' directly to several constructors.
    constructors = ("zeros", "eye", "tensor")
    for name in constructors:
        fn = getattr(torch, name)

        def cpu_constructor(*args, _fn=fn, **kwargs):
            if kwargs.get("device") == "cuda":
                kwargs["device"] = "cpu"
            return _fn(*args, **kwargs)

        monkeypatch.setattr(torch, name, cpu_constructor)
    reference = module.build_data(atoms, 12, dtype)
    data, metadata = build_scaling_data(atoms, 12, dtype, "cpu")
    assert tuple(data) == tuple(reference)
    for key in data:
        torch.testing.assert_close(data[key], reference[key], rtol=0, atol=0)
    assert metadata["n256_matches_previous_geometry_exactly"] == (atoms == 256)
    assert metadata["neighbors_per_atom"] == 12


def test_large_graph_uses_small_distance_search_only(monkeypatch):
    _template.cache_clear()
    shapes = []
    original_cdist = torch.cdist

    def cdist(left, right, *args, **kwargs):
        shapes.append((tuple(left.shape), tuple(right.shape)))
        assert left.shape[0] <= 256 and right.shape[0] <= 256
        return original_cdist(left, right, *args, **kwargs)

    monkeypatch.setattr(torch, "cdist", cdist)
    first, meta = build_scaling_data(1024, 12, torch.float64, "cpu")
    second, larger = build_scaling_data(4096, 12, torch.float64, "cpu")
    assert shapes == [((256, 3), (256, 3))]
    assert meta["motif_count"] == 4 and larger["motif_count"] == 16
    assert len(first["batch"].unique()) == 1
    assert torch.equal(first["edge_index"][0] // 256, first["edge_index"][1] // 256)
    assert first["edge_index"].shape == (2, 1024 * 12)
    assert first["edge_index"].min() == 0 and first["edge_index"].max() == 1023
    for data in (first, second):
        assert torch.equal(torch.bincount(data["edge_index"][1]), torch.full((len(data["species"]),), 12))
    # Fixed per-graph tensors are the only part that does not grow fourfold.
    keys = ("positions", "species", "edge_index", "edge_attr", "node_attrs", "edge_shifts", "batch")
    assert sum(second[k].numel() * second[k].element_size() for k in keys) == 4 * sum(
        first[k].numel() * first[k].element_size() for k in keys)
    assert meta["template_positions_and_edges_sha256"] == larger["template_positions_and_edges_sha256"]


def test_translated_motifs_keep_template_edges_and_relative_geometry():
    data, metadata = build_scaling_data(2048, 12, torch.float64, "cpu")
    template, _ = build_scaling_data(256, 12, torch.float64, "cpu")
    edge = data["edge_index"].reshape(2, 8, -1)
    offsets = torch.arange(8).mul(256)[None, :, None]
    assert torch.equal(edge - offsets, template["edge_index"][:, None, :].expand_as(edge))
    displacements = data["positions"][data["edge_index"][0]] - data["positions"][data["edge_index"][1]]
    baseline = template["positions"][template["edge_index"][0]] - template["positions"][template["edge_index"][1]]
    torch.testing.assert_close(displacements.reshape(8, -1, 3), baseline[None].expand(8, -1, -1), atol=4e-15, rtol=0)
    assert metadata["inter_motif_edges"] == 0
    assert metadata["graphs"] == 1 and data["ptr"].tolist() == [0, 2048]


@pytest.mark.parametrize("atoms,neighbors,dtype", [(12, 12, torch.float32), (257, 12, torch.float32),
                                                   (256, 0, torch.float32), (256, 12, torch.float16)])
def test_rejects_unsupported_geometry(atoms, neighbors, dtype):
    with pytest.raises(ValueError):
        build_scaling_data(atoms, neighbors, dtype, "cpu")
