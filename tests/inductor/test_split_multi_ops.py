# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the ``split_multi_ops`` IR pass."""

import unittest

import sympy
import torch
from torch import fx
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import FixedLayout, InputBuffer
from torch._inductor.virtualized import V

from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.split_multi_ops import _check_reproducible

_I0 = sympy.Symbol("_i0")


class TestSplitOfViewLoads(unittest.TestCase):
    """An intermediate over view loads is refused rather than miscompiled.

    split_multi_ops rebuilds each intermediate as an op over whole buffers, which
    reproduces a load only at its buffer's own position for the output shape.
    The traces below model a fused body over one output dim of 32 elements.
    """

    def setUp(self):
        gm = fx.symbolic_trace(lambda: None)
        self._graph_ctx = V.set_graph_handler(GraphLowering(gm))
        self._graph_ctx.__enter__()
        self.addCleanup(self._graph_ctx.__exit__, None, None, None)

    @staticmethod
    def _input(name, size):
        stride = [32, 1] if len(size) == 2 else [1]
        V.graph.name_to_buffer[name] = InputBuffer(
            name=name,
            layout=FixedLayout(torch.device("cpu"), torch.float32, size, stride),
        )

    @staticmethod
    def _load(vid, name, index):
        return ("load", vid, (), {"_name": name, "_index": index})

    def _check(self, trace):
        intermediate_ops = [e for e in trace if e[0] not in ("load", "store")][:-1]
        _check_reproducible(trace, intermediate_ops, [32], "buf0")

    def test_two_rows_of_one_buffer_are_refused(self):
        """Two select() rows of one buffer, combined, then converted."""
        self._input("arg", [3, 32])
        trace = [
            self._load(0, "arg", _I0),
            self._load(1, "arg", _I0 + 32),
            ("mul", 2, (0, 1), {}),
            ("to_dtype", 3, (2,), {}),
        ]
        with self.assertRaises(Unsupported):
            self._check(trace)

    def test_one_view_load_with_a_constant_is_allowed(self):
        """A single tensor operand is reloaded at its own index on replay."""
        self._input("arg", [3, 32])
        trace = [
            self._load(0, "arg", _I0 + 32),
            ("constant", 1, (), {"fill_value": 2.0, "dtype": torch.float32}),
            ("add", 2, (0, 1), {}),
            ("to_dtype", 3, (2,), {}),
        ]
        self._check(trace)

    def test_whole_buffers_are_allowed(self):
        """Two buffers read at their own positions combine correctly."""
        self._input("a", [32])
        self._input("b", [32])
        trace = [
            self._load(0, "a", _I0),
            self._load(1, "b", _I0),
            ("mul", 2, (0, 1), {}),
            ("to_dtype", 3, (2,), {}),
        ]
        self._check(trace)


if __name__ == "__main__":
    unittest.main()
