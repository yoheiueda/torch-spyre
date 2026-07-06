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


# transpose(0, -1).clone() swaps the OUTERMOST dim with the stick dim.  When both
# the source stick dim and the destination stick dim are sub-64 (e.g. 2 and 7),
# the restickify-padding candidate scan used to project the output stick coord
# through the input read dep, which composes the two sub-stick stride patterns
# into a multi-symbol coord and dropped the candidate -- so no fill was inserted
# and the restickify over-read uninitialized stick lanes (silent miscompile).
# Deriving the output stick dim from the output layout's own write dep fixes it.
SPLIT_T0_LAST = [(7, 67, 2), (7, 65, 2), (5, 3, 2), (7, 67, 63)]


@pytest.mark.parametrize("shape", SPLIT_T0_LAST, ids=lambda p: "x".join(map(str, p)))
def test_strict_transpose_0_last_clone(shape):
    x = _arange(*shape)
    _strict(lambda x: x.transpose(0, -1).clone(), x)


# Size-1 input-stick shapes ((7, 67, 1) etc.) are a distinct failure: upstream
# Inductor elides the size-1 source-stick dim (no loop symbol), so the
# restickify's input operand collapses to a 2-dim iteration space with no KERNEL
# data-stage and the backend aborts (dxp_standalone SIGABRT).  Two rewrites make
# N=1 match N>=2: superdsc._restore_elided_restickify_stick restores the elided
# stick as a fresh size-64 symbol so the SDSC descriptor is 3-dim, and the
# scheduler grows the output's collapsed size-1 device dim to a full stick so
# the physical allocation matches the 64-plane descriptor write
# (scheduler._grow_size1_stick_allocations).  Without the second fix the
# descriptor writes 64 planes into a 1-plane buffer and all but the first plane
# come back garbage.
#
# The .exp() makes the restickify input an internal ComputedBuffer, so
# insert_restickify_padding grows the producer in place (the fast path) rather
# than the zero-filled graph-input copy fallback.  A cheap arithmetic producer
# would be constant-folded away and never materialize the collapsed layout, so a
# transcendental is used; its last-ULP host/device drift means this asserts
# allclose rather than the exact torch.equal the ramp-based tests use.
SIZE1_INPUT_STICK = [(7, 67, 1), (7, 65, 1), (5, 3, 1)]


@pytest.mark.parametrize(
    "shape", SIZE1_INPUT_STICK, ids=lambda p: "x".join(map(str, p))
)
def test_size1_input_stick_transpose_0_last_clone(shape):
    def fn(x):
        return x.exp().transpose(0, -1).clone()

    x = torch.ones(*shape, dtype=torch.float16)
    spyre = _compile_and_run(fn, (x,), DEVICE)
    torch.testing.assert_close(spyre.cpu(), fn(x), atol=1e-2, rtol=1e-2)


# The bare clone (no .exp()) drives the same size-1 stick elision through the
# GRAPH-INPUT fallback: the restickify input is a graph input, so
# insert_restickify_padding takes the zero-filled copy path
# (_pad_restickify_input_via_copy) and rebuilds the restickify body via
# replace_computed_buffer_body -- which must carry the _size1_stick_alloc_dim tag
# onto the replacement buffer, or the scheduler grows nothing and the descriptor
# writes 64 planes into a 1-plane allocation (all but the first plane garbage).
# The copy path preserves the input bit-for-bit, so this asserts exact equality
# on a distinct ramp (unlike the .exp() fast path above).
@pytest.mark.parametrize(
    "shape", SIZE1_INPUT_STICK, ids=lambda p: "x".join(map(str, p))
)
def test_size1_input_stick_transpose_0_last_clone_graph_input(shape):
    x = _arange(*shape)
    _strict(lambda x: x.transpose(0, -1).clone(), x)


# >=2 size-1 host dims with a size-1 dim in the input stick.  The stick-dim
# projection (_host_dim_for_stick_sym) has no free symbol to match, so it takes
# the size-1 fallback -- and with several size-1 dims present that fallback picks
# the FIRST one rather than declining.  This is safe because size-1 dims do not
# contribute to the Spyre device layout (tensors_and_layouts.md canonical form):
# every size-1 host dim maps to host_size 1 and the physical dim to grow is
# re-derived from device-side stride_map markers, not this host index, so any
# size-1 dim yields the same layout.  These shapes assert that "pick the first"
# is byte-correct; the interleaved variants (size-1 dims not adjacent, a real dim
# between them) confirm the choice is independent of size-1 dim placement.  A
# genuine device-level ambiguity (>=2 size-1 *device* dims) still declines in
# _restickify_input_device_dim, so this fallback never masks a real hazard.
#
# Each entry is (shape, transpose_dims): the transpose must swap the size-1
# input-stick dim with a real dim, and the two dims not touched must both be
# size-1 (so >=2 size-1 dims and pick-first is exercised).
SIZE1_MULTI_STICK = [
    ((1, 1, 64, 1), (0, -1)),  # three size-1 dims (0, 1, 3)
    ((1, 1, 67, 1), (0, -1)),  # three size-1 dims, unaligned stick
    ((1, 5, 67, 1), (0, -1)),  # interleaved: size-1 at 0 and 3, real 5/67 between
    ((7, 1, 64, 1), (1, 3)),  # interleaved: size-1 at 1 and 3, real 7/64 between
    ((5, 1, 67, 1), (1, 3)),  # interleaved: size-1 at 1 and 3, real 5/67 between
]


@pytest.mark.parametrize(
    "shape,dims",
    SIZE1_MULTI_STICK,
    ids=[f"{'x'.join(map(str, s))}_t{d}" for s, d in SIZE1_MULTI_STICK],
)
def test_size1_multi_input_stick_transpose_clone(shape, dims):
    x = _arange(*shape)
    _strict(lambda x: x.transpose(*dims).clone(), x)


# A size-1 input-stick transpose where a real batch/leading dim (extent > 1)
# survives OUTSIDE both the old (size-1) and new sticks -- e.g. (4, 64, 1)
# transpose(1, 2), whose batch dim 0 stays leading while dims 1 and 2 swap.
# _restore_elided_restickify_stick used to reinsert the restored old stick
# immediately before the within-stick coord, but in the N>=2 descriptor the old
# stick lands at the rank the NEW stick occupies among the INPUT's coords -- the
# transpose swaps the two sticks' slots and every surviving dim keeps its place.
# When a batch dim sits between the two sticks those ranks differ, so the fixed
# "adjacent to within-stick" position mis-strided the batch dim and its non-first
# planes came back zeroed (all but batch plane 0 was garbage).  Deriving the
# insert position from the new stick's rank in the input coords fixes it.
#
# The new (destination) stick must be a full 64 here: an unaligned destination
# (e.g. 67) additionally needs output-middle stick padding, a separate path not
# covered by this restore-position fix.  Each entry is (shape, transpose_dims).
SIZE1_SURVIVING_BATCH = [
    ((4, 64, 1), (1, 2)),  # batch dim 0 = 4 survives; new stick = dim 1
    ((2, 64, 1), (1, 2)),  # smaller batch
    ((3, 5, 64, 1), (2, 3)),  # batch 3 + spatial 5 both survive; new stick dim 2
    ((2, 2, 64, 1), (2, 3)),  # two leading dims survive
    ((2, 3, 4, 64, 1), (3, 4)),  # deep batch nest
]


@pytest.mark.parametrize(
    "shape,dims",
    SIZE1_SURVIVING_BATCH,
    ids=[f"{'x'.join(map(str, s))}_t{d}" for s, d in SIZE1_SURVIVING_BATCH],
)
def test_size1_input_stick_surviving_batch_transpose_clone(shape, dims):
    x = _arange(*shape)
    _strict(lambda x: x.transpose(*dims).clone(), x)


# A size-1 input stick PLUS a second (leading or middle) size-1 host dim.  The
# transpose moves a real dim into the stick and demotes the size-1 stick to a
# collapsed non-stick device dim; the incidental extra size-1 dim ALSO collapses
# to a stride_map==-1 singleton, so the sole-``-1`` marker no longer isolates the
# old stick and _restickify_output_size1_device_dim must disambiguate by distance
# from the batch/preserved ("plane") dims (leading extra -> innermost grow dim,
# middle extra -> outermost).  Without that the collapsed old-stick alloc is
# never grown to a full stick and non-first batch planes come back zeroed
# (max_diff=255).  Baselines before the fix: the leading and batch-inner shapes
# below miscompiled; the middle / batch-outer ones already passed (kept as
# lock-in).  Distinct-ramp + torch.equal catches a mis-placed plane exactly.
SIZE1_EXTRA = [
    ((1, 4, 64, 1), (2, 3)),  # leading extra size-1 (dim0); old stick -> dim3
    ((4, 1, 64, 1), (2, 3)),  # middle extra size-1 (dim1); old stick -> dim0
    ((1, 4, 1, 64, 1), (3, 4)),  # two extra size-1, batch outer
    ((4, 1, 1, 64, 1), (3, 4)),  # two extra size-1, batch/size-1 interleaved
    ((1, 1, 4, 64, 1), (3, 4)),  # two extra size-1, batch inner
]


@pytest.mark.parametrize(
    "shape,dims",
    SIZE1_EXTRA,
    ids=[f"{'x'.join(map(str, s))}_t{d}" for s, d in SIZE1_EXTRA],
)
def test_size1_extra_dim_transpose_clone(shape, dims):
    x = _arange(*shape)
    _strict(lambda x: x.transpose(*dims).clone(), x)


# A size-1 input stick whose NEW stick dim spans >=2 stick blocks (host size
# > 64).  The elided size-1 old stick is restored by superdsc as a fresh 64-wide
# symbol; when the new stick is multi-block it splits into a tile-count device
# dim (floor(new_stick / 64)) that occupies the new stick's input rank, so the
# restored old stick must land one slot EARLIER (immediately outer to the block
# dim).  Inserting it at the block dim's slot gave the grown size-1 alloc a
# stride that collided with the block/batch host mapping and mis-placed the 2nd+
# stick block, so even a stick-ALIGNED multi-block new stick ([1,128,1]) came
# back wrong (max_diff=127).  Single-block new sticks are unaffected (their
# floor(.) slot is a degenerate extent-1 dim).  Distinct-ramp + torch.equal.
SIZE1_MULTI_BLOCK = [
    ((1, 128, 1), (1, 2)),  # 2 aligned blocks, no batch (was md=127)
    ((1, 192, 1), (1, 2)),  # 3 aligned blocks, no batch (was md=191)
    ((1, 67, 1), (1, 2)),  # 2 unaligned blocks, no batch (was md=66)
    ((4, 128, 1), (1, 2)),  # 2 aligned blocks + batch 4 (was md=511)
    ((2, 128, 1), (1, 2)),  # 2 aligned blocks + batch 2
    ((4, 67, 1), (1, 2)),  # 2 unaligned blocks + batch (was md=267)
    ((4, 192, 1), (1, 2)),  # 3 blocks + batch
    ((2, 3, 67, 1), (2, 3)),  # 2 unaligned blocks + leading batch nest (md=401)
]


@pytest.mark.parametrize(
    "shape,dims",
    SIZE1_MULTI_BLOCK,
    ids=[f"{'x'.join(map(str, s))}_t{d}" for s, d in SIZE1_MULTI_BLOCK],
)
def test_size1_multi_block_transpose_clone(shape, dims):
    x = _arange(*shape)
    _strict(lambda x: x.transpose(*dims).clone(), x)


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
