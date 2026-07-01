# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0
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
"""STRICT split-stick restickify + cat validation (distinct values + exact equality).

The shared ``compare_with_cpu`` uses ``atol=rtol=0.1`` on ``randn`` data, whose
fp16 value collisions in [-3, 3] can MASK a misplaced stick.  These tests use a
distinct-per-element ramp and ``torch.equal`` so any byte landing in the wrong
stick is caught.

Covers:
- ``transpose(...).clone()`` over split / unaligned stick dims, including the
  multi-batch shapes that used to drop inner batch planes (#1756).
- ``torch.cat`` on the Q2 target-model shapes from #1094, including mid-stick
  concats (split point not a stick multiple), which now compile to correct data
  via the reachable alt-layout selection + generalized restickify padding.
"""

import pytest
import torch
import torch_spyre  # noqa: F401
from utils_inductor import _compile_and_run, DEVICE


def _arange(*shape, base=0, span=1000):
    """A distinct-value ramp that is EXACT in fp16.

    Build the ramp in int64 and take the modulo BEFORE casting: fp16 cannot
    represent integers above 2048 exactly (``torch.arange`` itself overflows to
    inf past ~65504), and odd integers in (1024, 2048] are not representable, so
    a direct fp16 arange — or any band reaching past 1023 — silently rounds and
    turns the oracle's exact-equality check into a lie.

    ``base + span`` must stay <= 1024 so every value lands on an exact fp16
    integer (ULP < 1); a misplaced stick then shows up as a wrong value, never a
    rounding artifact.  ``base`` lets a second argument occupy a disjoint band
    from the first, so a swapped element is caught even between two cat inputs.
    """
    assert base + span <= 1024, f"band [{base}, {base + span}) exceeds fp16-exact 1024"
    n = 1
    for s in shape:
        n *= s
    ramp = (torch.arange(n, dtype=torch.int64) % span) + base
    return ramp.to(torch.float16).reshape(shape)


def _strict(fn, *args):
    spyre = _compile_and_run(fn, args, DEVICE)
    cpu = fn(*args)
    shapes = [tuple(a.shape) for a in args]
    assert torch.equal(spyre.cpu(), cpu), (
        f"\nMISMATCH shapes={shapes}\n cpu   =\n{cpu}\n spyre =\n{spyre.cpu()}\n"
    )


SPLIT_2D = [(65, 4), (67, 4), (128, 67), (130, 33)]


@pytest.mark.parametrize("shape", SPLIT_2D, ids=lambda p: f"{p[0]}x{p[1]}")
def test_strict_2d_transpose_clone(shape):
    x = _arange(*shape)
    _strict(lambda x: x.transpose(0, 1).clone(), x)


# transpose(-2, -1).clone() with >=2 leading non-degenerate batch dims used to
# drop inner batch planes.  Covers: single stick block (..64..), multi block
# (..65..), the size-4 old-stick middle dim that exposed both bailout guards in
# _restickify_output_middle_device_dim, and deeper/larger batch nests.
SPLIT_ND = [
    (4, 91, 72),
    (2, 3, 65, 4),
    (2, 3, 64, 4),
    (2, 2, 65, 64),
    (3, 5, 65, 4),
    (2, 4, 130, 33),
    (2, 2, 3, 65, 4),
]


@pytest.mark.parametrize("shape", SPLIT_ND, ids=lambda p: "x".join(map(str, p)))
def test_strict_nd_transpose_clone(shape):
    x = _arange(*shape)
    _strict(lambda x: x.transpose(-2, -1).clone(), x)


# transpose(1, -1).clone() swaps an inner batch dim with the stick dim, a
# different demoted-middle geometry than transpose(-2, -1).
SPLIT_T1_LAST = [(2, 4, 67, 128), (2, 3, 65, 4)]


@pytest.mark.parametrize("shape", SPLIT_T1_LAST, ids=lambda p: "x".join(map(str, p)))
def test_strict_nd_transpose_1_last_clone(shape):
    x = _arange(*shape)
    _strict(lambda x: x.transpose(1, -1).clone(), x)


# torch.cat shapes drawn from the Q2 target-model failures catalogued in
# issue #1094 (torch.cat for Ministral / Mistral-Small / gpt-oss-20b / granite).
# Each entry is (a_shape, b_shape, dim, marks).  These are the concrete shapes
# the model-enablement work must make correct.  A mid-stick concat (split point
# not a multiple of 64) relaid the cat-output buffer to a degenerate stick=0
# layout because _find_alt_target_stl picked the first offset-free candidate
# without checking it was reachable from the input stick; the cost model then
# rejected the real-stick->stick-0 restickify ("No mechanism to scatter elements
# from one stick to multiple sticks").  Fixed by ranking the alt layout by
# reachability (propagate_layouts._find_alt_target_stl) and generalizing the
# insert_restickify_padding perm loop to handle the size-1 host dims of these
# shapes; all cases below now produce correct data.
CAT_MODEL_SHAPES = [
    # cat_6 Ministral-3-14B: aligned 64+64, seq=1 — #1094 PASS.
    ((1, 8, 1, 64), (1, 8, 1, 64), -1, ()),
    # cat_1 Ministral-3-14B: aligned 64+64 (dense sticks) — #1094 PASS.
    ((1, 14, 64), (1, 14, 64), -1, ()),
    # cat_2 granite-3.3-8b: aligned 64+64 over a 4D shape — #1094 PASS.
    ((1, 32, 41, 64), (1, 32, 41, 64), -1, ()),
    # cat_4 granite-3.3-8b-fms: KV-cache append on seq dim 2, 67+1 -> 68.
    # #1094 reported FAIL; this branch produces correct data.
    ((1, 8, 67, 128), (1, 8, 1, 128), 2, ()),
    # cat_2 gpt-oss-20b: rotary embed, mid-stick 32+32 -> 64 — #1094 FAIL,
    # now fixed (reachable alt layout + generalized restickify-pad perm).
    ((1, 8, 11, 32), (1, 8, 11, 32), -1, ()),
    # cat_6 gpt-oss-20b: same rotary pattern, seq=1 — #1094 FAIL, now fixed.
    ((1, 8, 1, 32), (1, 8, 1, 32), -1, ()),
]


def _cat_id(p):
    a, b, dim, _ = p
    return f"{'x'.join(map(str, a))}+{'x'.join(map(str, b))}_dim{dim}"


@pytest.mark.parametrize(
    "shapes",
    [pytest.param(p, marks=p[3], id=_cat_id(p)) for p in CAT_MODEL_SHAPES],
)
def test_strict_cat_model_shapes(shapes):
    a_shape, b_shape, dim, _ = shapes
    # Put the two inputs in disjoint fp16-exact bands ([0, 512) and [512, 1024))
    # so a swapped element across the cat boundary is caught by torch.equal.
    x = _arange(*a_shape, base=0, span=512)
    y = _arange(*b_shape, base=512, span=512)
    _strict(lambda x, y: torch.cat([x, y], dim=dim), x, y)
