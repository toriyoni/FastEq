"""Frozen benchmark methods keep descriptors and zero-argument super semantics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "benchmarks/adapters"))
from v3_snapshot import _bind_methods


def test_snapshot_method_binding_descriptors_and_super():
    class Base:
        def label(self):
            return "base"

    class Current(Base):
        def helper(self):
            return "current"

    class Archived(Base):
        def helper(self):
            return "archived"

        def forward(self):
            return super().label(), self.helper()

        @staticmethod
        def static(value):
            return value + 3

        @classmethod
        def classmethod(cls):
            return cls

    instance = Current()
    _bind_methods(instance, Archived)
    assert instance.forward() == ("base", "archived")
    assert instance.static(2) == 5
    assert instance.classmethod() is Current
