"""O(N * neighbors) prebuilt geometry for the atom-count scaling experiment.

N=64/128 use the original geometry recipe. N=256 is bitwise identical to the
previous model benchmark. Larger sizes repeat that fixed 256-atom motif on a
three-dimensional grid. Motifs have no inter-motif edges but belong to one
graph; this is an operator/model capacity experiment, not a large-system MD
simulation. Neighbor construction is never part of the measured callable.
"""

import hashlib
import math
from functools import lru_cache

import torch

MOTIF_ATOMS = 256
MOTIF_SPACING = 8.0


@lru_cache(maxsize=4)
def _template(atoms, neighbors):
    """Only small CPU motifs use the old quadratic distance search."""
    generator = torch.Generator().manual_seed(2701 + atoms)
    positions = torch.rand(atoms, 3, generator=generator, dtype=torch.float64) * 5.0
    distances = torch.cdist(positions, positions)
    distances.fill_diagonal_(float("inf"))
    receiver = torch.arange(atoms).repeat_interleave(neighbors)
    sender = distances.topk(neighbors, largest=False).indices.flatten()
    edge_index = torch.stack((sender, receiver))
    sha = hashlib.sha256()
    sha.update(positions.numpy().tobytes())
    sha.update(edge_index.numpy().tobytes())
    lengths = (positions[sender] - positions[receiver]).norm(dim=1)
    metadata = {
        "template_atoms": atoms,
        "template_seed": 2701 + atoms,
        "template_positions_and_edges_sha256": sha.hexdigest(),
        "template_distance_dtype": "float64",
        "template_edge_distance_min": lengths.min().item(),
        "template_edge_distance_max": lengths.max().item(),
        "template_edge_distance_median": lengths.median().item(),
        "template_edges_within_cutoff_2": int((lengths < 2.0).sum()),
        "template_edges_within_cutoff_4": int((lengths < 4.0).sum()),
    }
    return positions, edge_index, metadata


def _grid_side(copies):
    side = max(1, math.ceil(copies ** (1.0 / 3.0)))
    while side ** 3 < copies:
        side += 1
    while side > 1 and (side - 1) ** 3 >= copies:
        side -= 1
    return side


def build_scaling_data(atoms, neighbors, dtype, device):
    """Return (old-model data dict, JSON metadata) without a large CPU copy.

    CUDA allocations are intentionally uncaught. The process-level worker
    records geometry-setup OOM separately from model-execution OOM. CPU is
    supported for small, deterministic geometry checks only.
    """
    if not isinstance(atoms, int) or atoms < neighbors + 1:
        raise ValueError("atoms must exceed neighbors")
    if not isinstance(neighbors, int) or not 0 < neighbors < MOTIF_ATOMS:
        raise ValueError("neighbors must be between 1 and 255")
    if atoms > MOTIF_ATOMS and atoms % MOTIF_ATOMS:
        raise ValueError("sizes above 256 must be multiples of the 256-atom motif")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("scaling supports only float32 and float64")
    device = torch.device(device)
    template_atoms = min(atoms, MOTIF_ATOMS)
    template_positions, template_edges, template_metadata = _template(template_atoms, neighbors)
    copies = atoms // template_atoms
    position_template = template_positions.to(device=device, dtype=dtype)
    edge_template = template_edges.to(device=device)
    if copies == 1:
        positions, edge_index = position_template.clone(), edge_template.clone()
        side = 1
    else:
        side = _grid_side(copies)
        motif_indices = torch.arange(copies, device=device)
        translations = torch.stack((motif_indices.remainder(side),
                                    motif_indices.div(side, rounding_mode="floor").remainder(side),
                                    motif_indices.div(side * side, rounding_mode="floor")), dim=1)
        translations = translations.to(dtype=dtype).mul_(MOTIF_SPACING)
        positions = position_template.unsqueeze(0).expand(copies, -1, -1).clone()
        positions.add_(translations.unsqueeze(1))
        positions = positions.reshape(atoms, 3)
        offsets = motif_indices.mul(MOTIF_ATOMS)
        edge_index = (edge_template.unsqueeze(1) + offsets[None, :, None]).reshape(2, -1)
    species = torch.arange(atoms, device=device).remainder(2)
    edges = atoms * neighbors
    data = {
        "positions": positions,
        "species": species,
        "edge_index": edge_index,
        "edge_attr": torch.zeros(edges, 3, device=device, dtype=dtype),
        "node_attrs": torch.nn.functional.one_hot(species, 2).to(dtype),
        "edge_shifts": torch.zeros(edges, 3, device=device, dtype=dtype),
        # No periodic edges. This preserves the previous input convention.
        "lattice": (torch.eye(3, device=device, dtype=dtype) * 20.0).unsqueeze(0),
        "batch": torch.zeros(atoms, dtype=torch.long, device=device),
        "ptr": torch.tensor([0, atoms], device=device),
        "fidelity_idx": torch.zeros(1, dtype=torch.long, device=device),
        "n_node": torch.tensor([atoms], device=device),
        "n_edge": torch.tensor([edges], device=device),
    }
    metadata = {
        "atoms": atoms,
        "edges": edges,
        "neighbors_per_atom": neighbors,
        "graphs": 1,
        "geometry": "original fixed geometry" if copies == 1 else "translated disconnected 256-atom motifs in one graph",
        "motif_count": copies,
        "grid_side": side,
        "motif_spacing": MOTIF_SPACING if copies > 1 else 0.0,
        "inter_motif_edges": 0,
        "prebuilt_neighbors": True,
        "neighbor_search_in_timing": False,
        "large_n_cpu_distance_matrix": False,
        "geometry_setup_complexity": "O(atoms * neighbors), after fixed <=256-atom CPU template",
        "n256_matches_previous_geometry_exactly": atoms == MOTIF_ATOMS,
        "input_tensor_logical_bytes": sum(value.numel() * value.element_size() for value in data.values()),
        "translation_rounding_note": "Each paired variant receives identical translated coordinates; FP32 subtraction can differ slightly from the unshifted motif.",
        **template_metadata,
    }
    return data, metadata
