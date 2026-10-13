"""Small, nonpersistent caches for immutable CUDA operands."""

from collections import OrderedDict

import torch


class TensorCache:
    def __init__(self, capacity=8, max_numel=1_048_576):
        self.capacity = capacity
        self.max_numel = max_numel
        self.values = OrderedDict()

    def get(self, key, factory):
        value = self.values.get(key)
        if value is not None:
            self.values.move_to_end(key)
            return value
        # A constant first created during inference must remain usable in a
        # subsequent training graph. Never cache a parameter or derived graph.
        with torch.inference_mode(False), torch.no_grad():
            value = factory()
        if value.numel() <= self.max_numel:
            self.values[key] = value
            if len(self.values) > self.capacity:
                self.values.popitem(last=False)
        return value


_indices = TensorCache()


def indices(x, n, *, sequence=False):
    return _indices.get(
        (x.device, n, sequence),
        lambda: (
            torch.arange(n, dtype=torch.long, device=x.device)
            if sequence
            else torch.zeros(n, dtype=torch.long, device=x.device)
        ),
    )
