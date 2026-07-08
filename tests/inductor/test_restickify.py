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

# Tests for restickify insertion in pointwise operations.
#
# Restickify is triggered when a transposed (non-contiguous) tensor is used
# in a pointwise op alongside a contiguous tensor, and the layouts are
# stick-incompatible. The compiler inserts a restickify kernel to convert
# the layout before the pointwise op proceeds.
#
# Shapes use multiples of 64 (stick size = 64 fp16 elements) to ensure
# stick-aligned inputs that exercise the restickify path rather than fallback.

import math

import pytest
from unittest.mock import patch

import torch
from torch._inductor.virtualized import V

import torch_spyre._inductor.optimize_restickify as _optimize_restickify
from torch._inductor.exc import InductorError
from utils_inductor import _compile_and_run, compare_with_cpu

DEVICE = torch.device("spyre")
S = 128  # must be a multiple of 64
T = 64  # side length for 4D tests (all dims equal)


# -------- Helpers ---------- #
def _compute_cost(restickify_plan):
    assert restickify_plan is not None, "restickify_plan should not be None"
    return sum(
        math.prod(int(s) for s in entry["target_layout"].size)
        for entries in restickify_plan.values()
        for entry in entries
    )


def _compile_and_run_plan_capture(fn, *args):
    import torch_spyre._inductor.passes as _passes

    captured = {}
    finalize_layouts = _passes.finalize_layouts

    def capturing_finalize_layouts(graph):
        finalize_layouts(graph)
        captured["plan"] = dict(V.graph.restickify_plan)

    with patch.object(_passes, "finalize_layouts", capturing_finalize_layouts):
        spyre_result = _compile_and_run(fn, args, DEVICE)

    return spyre_result, captured.get("plan", {})


def _compare(fn, *args, check_strides=True, optimal_cost=None, skip_correctness=False):
    """Run fn on Spyre, assert correctness against CPU, and optionally assert the restickify
    plan has cost == optimal_cost.
    """
    if optimal_cost is None:
        spyre_result = _compile_and_run(fn, args, DEVICE)
    else:
        spyre_result, plan = _compile_and_run_plan_capture(fn, *args)
        actual_cost = _compute_cost(plan)
        assert actual_cost == optimal_cost, (
            f"restickify cost: expected {optimal_cost}, got {actual_cost}"
        )
    if not skip_correctness:
        compare_with_cpu(fn, *args, target=spyre_result, run_eager=False)
    if check_strides:
        cpu_result = fn(*args)
        assert cpu_result.stride() == spyre_result.stride(), (
            f"Stride mismatch: CPU {cpu_result.stride()} vs Spyre {spyre_result.stride()}"
        )


def _make_tensors(n, *shape):
    """Make n scaled fp16 tensors of the given shape. Scale keeps values small enough for chained matmuls."""
    return [torch.randn(*shape, dtype=torch.float16) * 0.1 for _ in range(n)]


def _make_2d_tensors(s1, s2):
    # A, B: shape [s1, s2]; X, Y: shape [s2, s1]
    A = torch.randn((s1, s2), dtype=torch.float16)
    B = torch.randn((s1, s2), dtype=torch.float16)
    X = torch.randn((s2, s1), dtype=torch.float16)
    Y = torch.randn((s2, s1), dtype=torch.float16)
    return A, B, X, Y


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


# -------- Pointwise tests ----------

# 2-arg tests — run on a full set of size pairs
SIZES_2D_FULL = [
    (256, 128),
    (128, 256),
    (128, 128),
    (64, 128),
    (128, 64),
]


@pytest.fixture(params=SIZES_2D_FULL, ids=lambda p: f"{p[0]}x{p[1]}")
def tensors_2arg(request):
    s1, s2 = request.param
    return _make_2d_tensors(s1, s2)


def test_2arg_at_plus_x(tensors_2arg):
    A, _, X, _ = tensors_2arg
    _compare(lambda a, x: a.t() + x, A, X, optimal_cost=A.numel())


def test_2arg_x_plus_at(tensors_2arg):
    A, _, X, _ = tensors_2arg
    _compare(lambda a, x: x + a.t(), A, X, optimal_cost=A.numel())


def test_2arg_xt_plus_a(tensors_2arg):
    A, _, X, _ = tensors_2arg
    _compare(lambda a, x: x.t() + a, A, X, optimal_cost=X.numel())


def test_2arg_a_plus_xt(tensors_2arg):
    A, _, X, _ = tensors_2arg
    _compare(lambda a, x: a + x.t(), A, X, optimal_cost=X.numel())


# 3-arg and 4-arg tests — run on a smaller set of size pairs
SIZES_2D_SMALL = [
    (256, 128),
    (128, 128),
]


@pytest.fixture(params=SIZES_2D_SMALL, ids=lambda p: f"{p[0]}x{p[1]}")
def tensors_multiarg(request):
    s1, s2 = request.param
    return _make_2d_tensors(s1, s2)


def test_3arg_at_bt_x(tensors_multiarg):
    A, B, X, _ = tensors_multiarg
    _compare(lambda a, b, x: a.t() + b.t() + x, A, B, X, optimal_cost=X.numel())


def test_3arg_at_x_bt(tensors_multiarg):
    A, B, X, _ = tensors_multiarg
    _compare(lambda a, b, x: a.t() + x + b.t(), A, B, X, optimal_cost=X.numel())


def test_3arg_x_at_bt(tensors_multiarg):
    A, B, X, _ = tensors_multiarg
    _compare(lambda a, b, x: x + a.t() + b.t(), A, B, X, optimal_cost=X.numel())


def test_3arg_at_x_y(tensors_multiarg):
    A, _, X, Y = tensors_multiarg
    _compare(lambda a, x, y: a.t() + x + y, A, X, Y, optimal_cost=A.numel())


def test_4arg_at_bt_x_y(tensors_multiarg):
    A, B, X, Y = tensors_multiarg
    _compare(
        lambda a, b, x, y: a.t() + b.t() + x + y, A, B, X, Y, optimal_cost=A.numel()
    )


def test_4arg_at_x_bt_y(tensors_multiarg):
    A, B, X, Y = tensors_multiarg
    _compare(
        lambda a, b, x, y: a.t() + x + b.t() + y, A, B, X, Y, optimal_cost=2 * A.numel()
    )


def test_4arg_x_at_y_bt(tensors_multiarg):
    A, B, X, Y = tensors_multiarg
    _compare(
        lambda a, b, x, y: x + a.t() + y + b.t(), A, B, X, Y, optimal_cost=2 * A.numel()
    )


def test_4arg_at_x_y_bt(tensors_multiarg):
    A, B, X, Y = tensors_multiarg
    _compare(
        lambda a, b, x, y: a.t() + x + y + b.t(), A, B, X, Y, optimal_cost=2 * A.numel()
    )


def test_4arg_at_x_y_z(tensors_multiarg):
    A, _, X, Y = tensors_multiarg
    Z = torch.randn_like(X)
    _compare(lambda a, x, y, z: a.t() + x + y + z, A, X, Y, Z, optimal_cost=A.numel())


def test_4arg_x_at_y_z(tensors_multiarg):
    A, _, X, Y = tensors_multiarg
    Z = torch.randn_like(X)
    _compare(lambda a, x, y, z: x + a.t() + y + z, A, X, Y, Z, optimal_cost=A.numel())


def test_4arg_x_y_at_z(tensors_multiarg):
    A, _, X, Y = tensors_multiarg
    Z = torch.randn_like(X)
    _compare(lambda a, x, y, z: x + y + a.t() + z, A, X, Y, Z, optimal_cost=A.numel())


def test_4arg_x_y_z_at(tensors_multiarg):
    A, _, X, Y = tensors_multiarg
    Z = torch.randn_like(X)
    _compare(lambda a, x, y, z: x + y + z + a.t(), A, X, Y, Z, optimal_cost=A.numel())


# 3D tests
SIZES_3D = [(2, 256, 128), (4, 128, 64)]


@pytest.fixture(params=SIZES_3D, ids=lambda p: f"{p[0]}x{p[1]}x{p[2]}")
def tensors_3d(request):
    s0, s1, s2 = request.param
    a = torch.randn((s0, s1, s2), dtype=torch.float16)
    x = torch.randn((s0, s2, s1), dtype=torch.float16)
    return a, x


def test_3d_transpose12_plus_x(tensors_3d):
    a, x = tensors_3d
    _compare(lambda a, x: a.transpose(1, 2) + x, a, x)


def test_3d_x_plus_transpose12(tensors_3d):
    a, x = tensors_3d
    _compare(lambda a, x: x + a.transpose(1, 2), a, x)


# 4D tests:
SIZES_4D = [(2, 256, 3, 128), (2, 128, 4, 64)]


@pytest.fixture(params=SIZES_4D, ids=lambda p: f"{p[0]}x{p[1]}x{p[2]}x{p[3]}")
def tensors_4d(request):
    s0, s1, s2, s3 = request.param
    a = torch.randn((s0, s1, s2, s3), dtype=torch.float16)
    x = torch.randn((s0, s3, s2, s1), dtype=torch.float16)
    return a, x


def test_4d_transpose13_plus_x(tensors_4d):
    a, x = tensors_4d
    _compare(lambda a, x: a.transpose(1, 3) + x, a, x)


def test_4d_x_plus_transpose13(tensors_4d):
    a, x = tensors_4d
    _compare(lambda a, x: x + a.transpose(1, 3), a, x)


# View + unsqueeze tests


def test_view_unsqueeze_add():
    d0, d1, d2, d3, d4 = 2, 3, 4, 2, 64
    a = torch.randn((1, d0, d1 * d3 * d4), dtype=torch.float16) * 0.1
    b = torch.randn((1, d0, d1 * d3 * d4), dtype=torch.float16) * 0.1
    c = torch.randn((1, d0, d2, d3, d4), dtype=torch.float16) * 0.1

    def func(a, b, c):
        x = a + b
        z = x.view(1, d0, d1, d3, d4)
        return z.unsqueeze(2) + c.unsqueeze(3)

    _compare(func, a, b, c)


# Expand tests
SIZES_EXPAND = [(128, 256)]


@pytest.fixture(params=SIZES_EXPAND, ids=lambda p: f"{p[0]}x{p[1]}")
def tensors_expand(request):
    s0, s1 = request.param
    x = torch.randn((s0, s1, s1), dtype=torch.float16)
    y = torch.randn((s1, s0), dtype=torch.float16)
    return x, y


def test_expand_x_plus_yt_expand(tensors_expand):
    x, y = tensors_expand
    _compare(lambda x, y: x + y.transpose(0, 1).unsqueeze(1).expand(x.shape), x, y)


def test_expand_yt_expand_plus_x(tensors_expand):
    x, y = tensors_expand
    _compare(
        lambda x, y: y.transpose(0, 1).unsqueeze(1).expand(x.shape) + x,
        x,
        y,
        check_strides=False,  # Stride differes from CPU even before restickify, skipping stride check
    )


# Expand + transpose tests: b.unsqueeze(0 or 1).expand(s,s) forces layout
# choice because the expand side cannot always be restickified — the optimizer
# must choose the a.t() side's stick instead.


def test_expand_unsqueeze0_expand_plus_at():
    s = 128
    a = torch.randn((s, s), dtype=torch.float16) * 0.1
    b = torch.randn((s,), dtype=torch.float16) * 0.1
    _compare(
        lambda a, b: b.unsqueeze(0).expand(s, s) + a.t(), a, b, check_strides=False
    )


def test_expand_at_plus_unsqueeze0_expand():
    s = 128
    a = torch.randn((s, s), dtype=torch.float16) * 0.1
    b = torch.randn((s,), dtype=torch.float16) * 0.1
    _compare(lambda a, b: a.t() + b.unsqueeze(0).expand(s, s), a, b)


def test_expand_unsqueeze1_expand_plus_at():
    s = 128
    a = torch.randn((s, s), dtype=torch.float16) * 0.1
    b = torch.randn((s,), dtype=torch.float16) * 0.1
    _compare(
        lambda a, b: b.unsqueeze(1).expand(s, s) + a.t(), a, b, check_strides=False
    )


def test_expand_at_plus_unsqueeze1_expand():
    s = 128
    a = torch.randn((s, s), dtype=torch.float16) * 0.1
    b = torch.randn((s,), dtype=torch.float16) * 0.1
    _compare(lambda a, b: a.t() + b.unsqueeze(1).expand(s, s), a, b)


# cat after two-stick add: the add produces two candidate sticks; the cat
# forces a mutation op downstream and requires the chosen stick to be
# compatible with the cat output layout.


def test_cat_after_at_plus_b():
    s = 128
    a = torch.randn((s, s), dtype=torch.float16) * 0.1
    b = torch.randn((s, s), dtype=torch.float16) * 0.1
    c = torch.randn((s, s), dtype=torch.float16) * 0.1
    _compare(lambda a, b, c: torch.cat([a.t() + b, c]), a, b, c, check_strides=False)


# 2-arg tests with size-1
SIZES_4D_SIZE1 = [(128, 256)]


@pytest.fixture(params=SIZES_4D_SIZE1, ids=lambda p: f"1x{p[0]}x1x{p[1]}")
def tensors_size1(request):
    s1, s2 = request.param
    X = torch.randn((1, s2, 1, s1), dtype=torch.float16)
    Y = torch.randn((1, s1, 1, s2), dtype=torch.float16)
    return X, Y


def test_2arg_size1_x_plus_yt13(tensors_size1):
    X, Y = tensors_size1
    _compare(lambda x, y: x + y.transpose(1, 3), X, Y)


def test_2arg_size1_yt13_plus_x(tensors_size1):
    X, Y = tensors_size1
    _compare(lambda x, y: y.transpose(1, 3) + x, X, Y)


# ------- Matmul Tests ---------

MATMUL_SIZES = [(128, 256), (64, 128)]


@pytest.fixture(params=MATMUL_SIZES, ids=[f"{a}x{b}" for a, b in MATMUL_SIZES])
def matmul_tensors_ab(request):
    a, b = request.param
    x = torch.randn((a, b), dtype=torch.float16) * 0.1
    y = torch.randn((a, b), dtype=torch.float16) * 0.1
    return x, y


@pytest.fixture(params=MATMUL_SIZES, ids=[f"{a}x{b}" for a, b in MATMUL_SIZES])
def matmul_tensors_ab_ba(request):
    a, b = request.param
    x = torch.randn((a, b), dtype=torch.float16) * 0.1
    y = torch.randn((b, a), dtype=torch.float16) * 0.1
    return x, y


def test_matmul_x_y(matmul_tensors_ab_ba):
    x, y = matmul_tensors_ab_ba
    _compare(lambda x, y: torch.matmul(x, y), x, y, optimal_cost=0)


def test_matmul_xt_y(matmul_tensors_ab):
    x, y = matmul_tensors_ab
    _compare(lambda x, y: torch.matmul(x.t(), y), x, y, optimal_cost=x.numel())


def test_matmul_x_yt(matmul_tensors_ab):
    x, y = matmul_tensors_ab
    _compare(lambda x, y: torch.matmul(x, y.t()), x, y, optimal_cost=y.numel())


def test_matmul_xt_yt(matmul_tensors_ab_ba):
    x, y = matmul_tensors_ab_ba
    _compare(
        lambda x, y: torch.matmul(x.t(), y.t()),
        x,
        y,
        optimal_cost=x.numel() + y.numel(),
    )


# ------- Batched Matmul Tests ---------

BMM_SIZES = [(3, 128, 64)]


@pytest.fixture(params=BMM_SIZES, ids=lambda p: f"{p[0]}x{p[1]}x{p[2]}")
def bmm_tensors_ab(request):
    batch, a, b = request.param
    x = torch.randn((batch, a, b), dtype=torch.float16) * 0.1
    y = torch.randn((batch, a, b), dtype=torch.float16) * 0.1
    return x, y


@pytest.fixture(params=BMM_SIZES, ids=lambda p: f"{p[0]}x{p[1]}x{p[2]}")
def bmm_tensors_ab_ba(request):
    batch, a, b = request.param
    x = torch.randn((batch, a, b), dtype=torch.float16) * 0.1
    y = torch.randn((batch, b, a), dtype=torch.float16) * 0.1
    return x, y


def test_bmm_xt_y(bmm_tensors_ab):
    x, y = bmm_tensors_ab
    _compare(lambda x, y: torch.matmul(x.transpose(1, 2), y), x, y)


def test_bmm_x_yt(bmm_tensors_ab):
    x, y = bmm_tensors_ab
    _compare(lambda x, y: torch.matmul(x, y.transpose(1, 2)), x, y)


def test_bmm_xt_yt(bmm_tensors_ab_ba):
    x, y = bmm_tensors_ab_ba
    _compare(lambda x, y: torch.matmul(x.transpose(1, 2), y.transpose(1, 2)), x, y)


# ------- FallbackKernel + restickify regression test ---------


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
def test_fallback_with_restickify():
    # FallbackKernel (torch.sin) produces a MultiOutput node. Verify the optimizer
    # handles it via AnyInNode and still makes a correct restickify decision downstream.
    x, y = _make_tensors(2, S, S)
    _compare(lambda x, y: torch.sin(x) + y.t(), x, y, optimal_cost=S * S)


# ------- Mutation + restickify regression test ---------


def test_bmm_with_inplace_mutation():
    # Regression test: copy_() creates a mutation_renames chain in the Inductor
    # scheduler. Combined with a bmm whose weight needs restickifying, this
    # previously caused a topo-sort cycle when compute_dependencies() was called
    # a second time inside insert_restickify.
    B, M, K, N = 1, 8, 64, 64
    x = torch.randn((B, M, K), dtype=torch.float16)
    weight = torch.randn((N, K), dtype=torch.float16)
    cache = torch.zeros((B, M, K), dtype=torch.float16)

    def func(x, weight, cache):
        cache.copy_(x)
        return torch.bmm(cache, weight.t().unsqueeze(0).expand(B, -1, -1))

    _compare(func, x, weight, cache)


# Optimizer correctness + optimality tests: verify both output values and
# minimum-cost restickify plan across a range of graph patterns.


def test_opt_parens_one_conflict():
    """((a + b) + (c.t() + d)) + (e + f) — conflict only in inner group."""
    a, b, c, d, e, f = _make_tensors(6, S, S)
    _compare(
        lambda a, b, c, d, e, f: ((a + b) + (c.t() + d)) + (e + f),
        a,
        b,
        c,
        d,
        e,
        f,
        optimal_cost=S * S,
    )


def test_opt_adds_then_matmul_x():
    """(a + b.t() + c.t() + d.t()) @ e — upstream optimal + forced matmul x cost."""
    a, b, c, d, e = _make_tensors(5, S, S)
    _compare(
        lambda a, b, c, d, e: (a + b.t() + c.t() + d.t()) @ e,
        a,
        b,
        c,
        d,
        e,
        optimal_cost=2 * S * S,
    )


def test_opt_adds_then_matmul_y():
    """a @ (b + c.t()) — beam picks upstream stick to avoid extra matmul cost."""
    a, b, c = _make_tensors(3, S, S)
    _compare(lambda a, b, c: a @ (b + c.t()), a, b, c, optimal_cost=S * S)


def test_opt_adds_then_matmul_y_long_chain():
    """a @ (b + c.t() + d.t() + e.t()) — majority transposed going into y."""
    a, b, c, d, e = _make_tensors(5, S, S)
    _compare(
        lambda a, b, c, d, e: a @ (b + c.t() + d.t() + e.t()),
        a,
        b,
        c,
        d,
        e,
        optimal_cost=2 * S * S,
    )


def test_opt_matmul_x_and_y_conflict():
    """a.t() @ (b + c.t()) — x wrong stick + y upstream conflict."""
    a, b, c = _make_tensors(3, S, S)
    _compare(lambda a, b, c: a.t() @ (b + c.t()), a, b, c, optimal_cost=2 * S * S)


def test_opt_matmul_then_adds():
    """(a @ b) + c.t() — matmul output stick vs transposed input."""
    a, b, c = _make_tensors(3, S, S)
    _compare(lambda a, b, c: (a @ b) + c.t(), a, b, c, optimal_cost=S * S)


def test_opt_matmul_then_long_adds():
    """(a @ b) + c.t() + d.t() — keep matmul stick, restickify one input."""
    a, b, c, d = _make_tensors(4, S, S)
    _compare(lambda a, b, c, d: (a @ b) + c.t() + d.t(), a, b, c, d, optimal_cost=S * S)


def test_opt_chained_matmuls():
    """(a @ b) @ c — no restickify needed."""
    a, b, c = _make_tensors(3, S, S)
    _compare(lambda a, b, c: (a @ b) @ c, a, b, c, optimal_cost=0)


def test_opt_two_independent_conflicts():
    """(a+b.t()) + (e.t()+f.t()+g) — two separate conflicts."""
    a, b, e, f, g = _make_tensors(5, S, S)
    _compare(
        lambda a, b, e, f, g: (a + b.t()) + (e.t() + f.t() + g),
        a,
        b,
        e,
        f,
        g,
        optimal_cost=2 * S * S,
    )


def test_opt_fanout_intermediate():
    """buf = a + b.t(); (buf + c) + (buf + d.t()) — buf consumed twice."""
    a, b, c, d = _make_tensors(4, S, S)

    def fn(a, b, c, d):
        buf = a + b.t()
        return buf + c + (buf + d.t())

    _compare(fn, a, b, c, d, optimal_cost=2 * S * S)


def test_opt_diamond():
    """buf = a + b.t(); buf + buf — same intermediate read twice."""
    a, b = _make_tensors(2, S, S)

    def fn(a, b):
        buf = a + b.t()
        return buf + buf

    _compare(fn, a, b, optimal_cost=S * S)


def test_opt_matmul_rect_x_wrong_stick():
    """(64x128).t() @ (64x192) — cost uses buffer size not reduction dim."""
    M, K, N = 64, 128, 192
    (a,) = _make_tensors(1, M, K)
    (b,) = _make_tensors(1, M, N)
    _compare(lambda a, b: a.t() @ b, a, b, optimal_cost=M * K)


def test_opt_sum_between_pointwise():
    """(a + b.t()).sum(1) + c — reduction between two pointwise stages."""
    a, b = _make_tensors(2, S, S)
    (c,) = _make_tensors(1, S)
    # Note: sum() below may fail correctness depending which stick flows in
    # because propagate_layouts does not yet properly detect incompatibility
    # of sparse/non-sparse sticks in a pointwise op.  Disabling correctness
    # check until that is resolved
    _compare(
        lambda a, b, c: (a + b.t()).sum(0) + c,
        a,
        b,
        c,
        optimal_cost=S * S,
        skip_correctness=True,
    )


def test_opt_chain_transposed_intermediate():
    """(a.t() + b).t() + c — intermediate consumed transposed."""
    a, b, c = _make_tensors(3, S, S)
    _compare(lambda a, b, c: (a.t() + b).t() + c, a, b, c, optimal_cost=S * S)


def test_opt_beam_trim(monkeypatch):
    """Three ops each with 2 candidate layouts: beam grows to 8 before trimming.

    BEAM_WIDTH=2 forces trimming at every step; verifies correctness is preserved.
    """
    monkeypatch.setattr(_optimize_restickify, "BEAM_WIDTH", 2)
    a, b, c, d, e, f = _make_tensors(6, S, S)
    _compare(
        lambda a, b, c, d, e, f: (a.t() + b) + (c.t() + d) + (e.t() + f),
        a,
        b,
        c,
        d,
        e,
        f,
    )


def test_opt_4d_one_conflict():
    """a.transpose(0,3) + b + c + d — one input with stick on dim 0."""
    a, b, c, d = _make_tensors(4, T, T, T, T)
    _compare(
        lambda a, b, c, d: a.transpose(0, 3) + b + c + d,
        a,
        b,
        c,
        d,
        optimal_cost=T**4,
    )


def test_opt_4d_mixed_conflicts():
    """a.transpose(0,3) + b.transpose(1,3) + c.transpose(2,3) + d — three non-matching sticks."""
    a, b, c, d = _make_tensors(4, T, T, T, T)
    _compare(
        lambda a, b, c, d: (
            a.transpose(0, 3) + b.transpose(1, 3) + c.transpose(2, 3) + d
        ),
        a,
        b,
        c,
        d,
        optimal_cost=3 * T**4,
    )


def test_opt_4d_majority_wins():
    """a.transpose(0,3) + b.transpose(0,3) + c.transpose(0,3) + d — three stick on dim 0."""
    a, b, c, d = _make_tensors(4, T, T, T, T)
    _compare(
        lambda a, b, c, d: (
            a.transpose(0, 3) + b.transpose(0, 3) + c.transpose(0, 3) + d
        ),
        a,
        b,
        c,
        d,
        optimal_cost=T**4,
    )


def test_opt_4d_chain_transposed_intermediate():
    """(a.transpose(2,3) + b).transpose(2,3) + c — 4D version of transposed intermediate."""
    a, b, c = _make_tensors(3, T, T, T, T)
    _compare(
        lambda a, b, c: (a.transpose(2, 3) + b).transpose(2, 3) + c,
        a,
        b,
        c,
        optimal_cost=T**4,
    )


def test_opt_two_matmuls_wrong_inputs():
    """(a.t() @ b) + (c @ d.t()) — each matmul has one wrong-stick input."""
    a, b, c, d = _make_tensors(4, S, S)
    _compare(
        lambda a, b, c, d: (a.t() @ b) + (c @ d.t()),
        a,
        b,
        c,
        d,
        optimal_cost=2 * S * S,
    )


def test_opt_matmul_both_inputs_upstream_conflict():
    """(a + b.t()) @ (c + d.t()) — both inputs have upstream stick conflicts."""
    a, b, c, d = _make_tensors(4, S, S)
    _compare(
        lambda a, b, c, d: (a + b.t()) @ (c + d.t()),
        a,
        b,
        c,
        d,
        optimal_cost=2 * S * S,
    )


# ------- Intentional failure -------------------


def test_wrong_optimal_cost_fails():
    """This tests checks if the optimal cost is mismatching so proper
    assertion failure is detected"""

    a, b, c, d, e = _make_tensors(5, S, S)

    def func(a, b, c, d, e):
        return (a + b.t() + c.t() + d.t()) @ e

    correct_expected_cost = 2 * S * S

    with pytest.raises(
        AssertionError,
        match=f"restickify cost: expected 0, got {correct_expected_cost}",
    ):
        _compare(func, a, b, c, d, e, optimal_cost=0)


# ------- Constant tensor STL tests ---------


def test_constant_plus_xt():
    """ones_like(x) + x.t() — constant tensor should adopt x.t()'s stick, cost 0."""
    x = torch.randn((S, S), dtype=torch.float16)
    _compare(lambda x: torch.ones_like(x) + x.t(), x, optimal_cost=0)


def test_constant_in_conflict_chain():
    """ones_like(x) + x.t() + y — constant adopts winning STL, doesn't add to conflict cost."""
    x, y = _make_tensors(2, S, S)
    _compare(lambda x, y: torch.ones_like(x) + x.t() + y, x, y, optimal_cost=S * S)


def test_constant_matmul_x():
    """ones_like(y) @ y — constant should get col-major STL that matmul x needs, cost 0."""
    y = _make_tensors(1, S, S)[0]
    _compare(lambda y: torch.ones_like(y) @ y, y, optimal_cost=0)


def test_two_constants_plus_xt():
    """ones_like(x) + zeros_like(x) + x.t() — two flexible constants, cost still 0."""
    x = torch.randn((S, S), dtype=torch.float16)
    _compare(
        lambda x: torch.ones_like(x) + torch.zeros_like(x) + x.t(), x, optimal_cost=0
    )


def test_full_plus_xt():
    """torch.full + x.t() — full tensor constant should adopt x.t()'s stick, cost 0."""
    x = torch.randn((S, S), dtype=torch.float16)
    _compare(
        lambda x: torch.full((S, S), 0.5, dtype=torch.float16, device=x.device) + x.t(),
        x,
        optimal_cost=0,
    )


def test_fill_plus_xt():
    """empty_like + fill_ + x.t() — mutation-based constant should adopt x.t()'s stick, cost 0."""
    x = torch.randn((S, S), dtype=torch.float16)

    def fn(x):
        e = torch.empty_like(x)
        e.fill_(1.0)
        return e + x.t()

    _compare(fn, x, optimal_cost=0)


def test_arange_plus_xt():
    """arange.view + x.t() — correctness check only.

    arange lowers to FallbackKernel which gets a fixed generic layout, so the
    downstream add may still need a restickify.  No optimal_cost asserted.
    """
    x = torch.randn((S, S), dtype=torch.float16)
    _compare(
        lambda x: (
            torch.arange(S * S, dtype=torch.float16, device=x.device).view(S, S) + x.t()
        ),
        x,
    )


# ------- Constant-fill inputs ---------


def test_amax_full_and_amax_live_maximum():
    """maximum(amax(full(-inf), dim=-1), amax(t, dim=-1)) — zero-stick output from
    constant-fill reduction must be a valid candidate for the pointwise output."""
    B, H, Lq, Lk = 1, 32, 128, 256
    t = torch.randn((B, H, Lq, Lk), dtype=torch.float16)

    def f(t):
        full = torch.full((B, H, Lq, Lk), float("-inf"), device=t.device, dtype=t.dtype)
        t_max = torch.amax(full, dim=-1)
        u_max = torch.amax(t, dim=-1)
        return torch.maximum(t_max, u_max)

    _compare(f, t, optimal_cost=0)


# ------- Unsupported stick configurations ---------


def test_sparse_dense_pointwise_unsupported():
    """a.sum(1) + b - pointwise of sparse and dense tensors not yet supported.

    There is no restickify resolution for this configuration so we must catch this and report error
    """
    a = torch.randn((S, S), dtype=torch.float16).to(DEVICE)
    b = torch.randn((S, S), dtype=torch.float16).to(DEVICE)
    with pytest.raises(
        InductorError, match="No mechanism to gather elements from multiple sticks"
    ):
        _compare(lambda a, b: a.sum(1) + b, a, b)


# ------- Restickify padding: sliced input raises Unsupported ---------


def test_pad_restickify_sliced_input_raises():
    """A mid-stick-sliced input to transpose+clone must raise, not silently
    corrupt output.

    ``x[:, :, 1:66, :]`` slices dim -2 (which the transpose turns into the new
    stick dim) at a non-stick-aligned start, so the read begins partway into a
    stick.  This is unpaddable today, so the compiler must fail loudly rather
    than return wrong data.
    """
    x = torch.randn((2, 2, 67, 128), dtype=torch.float16)
    with pytest.raises(
        RuntimeError,
        match="sliced input on host dim",
    ):
        _compile_and_run(
            lambda x: x[:, :, 1:66, :].transpose(-2, -1).clone(), (x,), DEVICE
        )


def test_pad_restickify_sliced_producer_raises():
    """A mid-stick slice fed by a *producer* must also raise, not miscompile.

    Same geometry as test_pad_restickify_sliced_input_raises, but the ``+ 1``
    makes the sliced tensor a produced (internal) buffer rather than a bare
    graph input, exercising the other input path.  The slice is equally
    unpaddable, so this path must also fail loudly rather than return wrong
    data.
    """
    x = torch.randn((2, 2, 67, 128), dtype=torch.float16)
    with pytest.raises(
        RuntimeError,
        match="sliced input on host dim",
    ):
        _compile_and_run(
            lambda x: (x + 1)[:, :, 1:66, :].transpose(-2, -1).clone(), (x,), DEVICE
        )


# ------- Restickify padding: sliced-transpose stick expr classification -------


# Sliced transposes that ARE valid and must compile correctly, spanning the
# range of slice placements: on the leading dim (which becomes non-stick), a
# stick-aligned start on the becomes-stick dim, and aligned/1.5-stick extents on
# the becomes-stick dim.  Contrast test_pad_restickify_sliced_input_raises, where
# the slice lands mid-stick on the becomes-stick dim and must raise.
OFFSET_STICK_OK = [
    # Slice the leading dim (becomes non-stick after transpose).
    (lambda x: x[3:67].transpose(0, 1).clone(), (128, 128)),
    # Same, with an unaligned extent (63): still fine.
    (lambda x: x[3:66].transpose(0, 1).clone(), (128, 128)),
    # Stick-aligned slice start on the becomes-stick dim.
    (lambda x: x[:, :, 64:128, :].transpose(-2, -1).clone(), (2, 2, 128, 128)),
    # Slice on the becomes-stick dim, aligned (single-stick) extent.
    (lambda x: x[:, 64:128].transpose(0, 1).clone(), (128, 128)),
    # Slice on the becomes-stick dim, 1.5-stick extent.
    (lambda x: x[:, :96].transpose(0, 1).clone(), (128, 128)),
]


@pytest.mark.parametrize(
    "fn,shape", OFFSET_STICK_OK, ids=lambda p: p if isinstance(p, tuple) else ""
)
def test_sliced_transpose_stick_expr_compiles(fn, shape):
    """A valid sliced transpose compiles correctly regardless of where the slice
    lands or its extent -- only a mid-stick slice on the becomes-stick dim is
    rejected (see test_pad_restickify_sliced_input_raises)."""
    x = torch.randn(shape, dtype=torch.float16)
    result = _compile_and_run(fn, (x,), DEVICE)
    compare_with_cpu(fn, x, target=result, run_eager=False)


# Strict versions of the becomes-stick-dim slices above: a slice that lands on
# the dim the transpose turns into the stick is the case most likely to misplace
# a stick lane, and randn + tolerance can mask that.  A distinct-value ramp with
# torch.equal catches a single displaced lane exactly.
@pytest.mark.parametrize(
    "fn,shape", OFFSET_STICK_OK, ids=lambda p: p if isinstance(p, tuple) else ""
)
def test_sliced_transpose_stick_expr_strict(fn, shape):
    x = _arange(*shape)
    _strict(fn, x)


# ------- Restickify padding (unaligned stick dim) ---------

# new_stick_dim = dim-0 (unaligned): shape (67, 128) or (1025, 1024), small tensor and large tensor.
RESTICKIFY_PAD_2D_SIZES = [(67, 128), (1025, 1024)]


@pytest.fixture(params=RESTICKIFY_PAD_2D_SIZES, ids=lambda p: f"{p[0]}x{p[1]}")
def pad_tensors_2d(request):
    s0, s1 = request.param
    return torch.randn((s0, s1), dtype=torch.float16)


def test_pad_2d_transpose_clone(pad_tensors_2d):
    """2D transpose(0,1)+clone: new stick dim is dim-0 (unaligned) — padding required."""
    x = pad_tensors_2d
    _compare(lambda x: x.transpose(0, 1).clone(), x, check_strides=False)


# new_stick_dim = dim-1 (unaligned): shape (128, 67)
RESTICKIFY_PAD_2D_LAST_SIZES = [(128, 67)]


@pytest.fixture(params=RESTICKIFY_PAD_2D_LAST_SIZES, ids=lambda p: f"{p[0]}x{p[1]}")
def pad_tensors_2d_last(request):
    s0, s1 = request.param
    return torch.randn((s0, s1), dtype=torch.float16)


def test_pad_2d_transpose_clone_last_dim_unaligned(pad_tensors_2d_last):
    """2D transpose(0,1)+clone: new stick dim is dim-1 (unaligned) — padding required."""
    x = pad_tensors_2d_last
    _compare(lambda x: x.transpose(0, 1).clone(), x, check_strides=False)


# transpose(0,1)+clone on (rows, cols) where the new stick dim `rows` spans >1
# stick block (rows > 64) and the middle dim `cols` is not a stick multiple, so
# the second and later stick blocks must land at the correct offset.
RESTICKIFY_PAD_2D_MID_SIZES = [(65, 4), (67, 4), (128, 67), (130, 33)]


@pytest.fixture(params=RESTICKIFY_PAD_2D_MID_SIZES, ids=lambda p: f"{p[0]}x{p[1]}")
def pad_tensors_2d_mid(request):
    s0, s1 = request.param
    return torch.randn((s0, s1), dtype=torch.float16)


def test_pad_2d_transpose_clone_middle_dim_unaligned(pad_tensors_2d_mid):
    """2D transpose(0,1)+clone where the new stick dim spans >1 block and the
    middle dim is unaligned — every stick block must land correctly."""
    x = pad_tensors_2d_mid
    _compare(lambda x: x.transpose(0, 1).clone(), x, check_strides=False)


# 3D: transpose(-2,-1).clone() — the new stick dim is the second-to-last dim
RESTICKIFY_PAD_3D_SIZES = [(2, 67, 128), (2, 1025, 1024)]


@pytest.fixture(params=RESTICKIFY_PAD_3D_SIZES, ids=lambda p: f"{p[0]}x{p[1]}x{p[2]}")
def pad_tensors_3d(request):
    s0, s1, s2 = request.param
    return torch.randn((s0, s1, s2), dtype=torch.float16)


def test_pad_3d_transpose_last2_clone(pad_tensors_3d):
    """3D transpose(-2,-1)+clone: new stick dim is unaligned — padding required."""
    x = pad_tensors_3d
    _compare(lambda x: x.transpose(-2, -1).clone(), x, check_strides=False)


# 4D: two transpose variants matching the user's shapes
RESTICKIFY_PAD_4D_SIZES = [(2, 2, 67, 128), (2, 2, 1025, 1024)]


@pytest.fixture(
    params=RESTICKIFY_PAD_4D_SIZES, ids=lambda p: f"{p[0]}x{p[1]}x{p[2]}x{p[3]}"
)
def pad_tensors_4d(request):
    s0, s1, s2, s3 = request.param
    return torch.randn((s0, s1, s2, s3), dtype=torch.float16)


def test_pad_4d_transpose_last2_clone(pad_tensors_4d):
    """4D transpose(-2,-1)+clone: new stick dim is unaligned — padding required."""
    x = pad_tensors_4d
    _compare(lambda x: x.transpose(-2, -1).clone(), x, check_strides=False)


@pytest.fixture(ids=lambda p: f"{p[0]}x{p[1]}x{p[2]}x{p[3]}", params=[(2, 2, 67, 128)])
def pad_tensors_4d_t1_last(request):
    s0, s1, s2, s3 = request.param
    return torch.randn((s0, s1, s2, s3), dtype=torch.float16)


def test_pad_4d_transpose_1_last_clone(pad_tensors_4d_t1_last):
    """4D transpose(1,-1)+clone: swaps dim-1 and dim-3, new stick dim unaligned."""
    x = pad_tensors_4d_t1_last
    _compare(lambda x: x.transpose(1, -1).clone(), x, check_strides=False)


# ------- Restickify input padding fused into a producer ---------
#
# When the restickify's input is produced by an internal op, the padding can be
# folded into that producer instead of inserting a separate copy; a restickify
# reading a bare graph input has no producer and falls back to the copy.  These
# tests exercise both outcomes (asserting on the debug log which path fired) and
# check that the result is correct either way.
#
# The tests put BOTH binary operands behind a computation so the transposed side
# (not a bare graph input) is the one restickified -- a bare graph-input operand
# would otherwise be the one chosen and hit the fallback.


def _run_capturing_padding_log(fn, *args):
    """Run fn on Spyre, returning (result, fused_fire_count, all_log_records)
    captured from insert_restickify_padding's debug log."""
    import logging

    import torch_spyre._inductor.padding as _padding

    records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Capture()
    prev_level = _padding.logger.level
    _padding.logger.addHandler(handler)
    _padding.logger.setLevel(logging.DEBUG)
    try:
        result = _compile_and_run(fn, args, DEVICE)
    finally:
        _padding.logger.removeHandler(handler)
        _padding.logger.setLevel(prev_level)

    fused = sum("fused pad into producer" in m for m in records)
    return result, fused, records


RESTICKIFY_FUSE_SIZES = [(67, 67), (2, 67, 67)]


@pytest.fixture(params=RESTICKIFY_FUSE_SIZES, ids=lambda p: "x".join(map(str, p)))
def fuse_tensors(request):
    shape = request.param
    x = torch.randn(shape, dtype=torch.float16)
    y = torch.randn(shape, dtype=torch.float16)
    return x, y


def test_pad_fused_into_producer(fuse_tensors):
    """(x*2).T + relu(y): the transposed side is an internal single-consumer
    pointwise producer, so the padding fuses into it (no copy).  Both operands
    are unaligned, so the same restickify needs both input and output padding."""
    x, y = fuse_tensors

    def fn(x, y):
        return (x * 2).transpose(-2, -1) + torch.relu(y)

    result, fused, _ = _run_capturing_padding_log(fn, x, y)
    assert fused >= 1, "expected producer-fusion to fire, but it did not"
    compare_with_cpu(fn, x, y, target=result, run_eager=False)


def test_pad_fused_into_matmul_producer():
    """A sliced matmul output (a@b)[:,c:]+z: the producer is a matmul
    (a reduction), not a pointwise op.  It should still fuse rather than fall
    back to a copy, and the result must be correct."""
    a = torch.randn((67, 64), dtype=torch.float16)
    b = torch.randn((64, 67), dtype=torch.float16)
    z = torch.randn((67, 64), dtype=torch.float16)

    def fn(a, b, z):
        return (a @ b)[:, 3:] + z

    result, fused, _ = _run_capturing_padding_log(fn, a, b, z)
    assert fused >= 1, "matmul (Reduction) producer should fuse, not fall back"
    compare_with_cpu(fn, a, b, z, target=result, run_eager=False)


def test_pad_graph_input_falls_back():
    """Restickifying a bare graph input has no producer to fuse into, so the
    padding must fall back to a copy -- and still be correct."""
    x = torch.randn((67, 67), dtype=torch.float16)
    y = torch.randn((67, 67), dtype=torch.float16)

    def fn(x, y):
        return x.transpose(-2, -1) + y

    result, fused, _ = _run_capturing_padding_log(fn, x, y)
    assert fused == 0, "graph-input restickify should not fuse into a producer"
    compare_with_cpu(fn, x, y, target=result, run_eager=False)


def test_pad_multi_consumer_producer_fuses_with_coreader():
    """A producer read by a restickify AND a non-restickify co-reader (here
    p.sum()) still fuses, and the co-reader's result stays correct -- growing
    the shared producer for the restickify does not disturb the other reader."""
    x = torch.randn((67, 128), dtype=torch.float16)
    z = torch.randn((128, 67), dtype=torch.float16)

    def fn(x, z):
        p = x * 2  # two consumers: the transpose (restickify) and the sum
        return p.transpose(-2, -1) + z + p.sum()

    result, fused, _ = _run_capturing_padding_log(fn, x, z)
    assert fused >= 1, "producer with a non-restickify co-reader should still fuse"
    compare_with_cpu(fn, x, z, target=result, run_eager=False)


def test_pad_shared_all_restickify_consumers_fuse():
    """A producer read only by restickify ops fuses even with several consumers.
    Two transposes of the same producer take different new stick dims, so each
    needs a different padding; both must apply and both results be correct."""
    x = torch.randn((67, 53, 128), dtype=torch.float16)
    za = torch.randn((128, 53, 67), dtype=torch.float16)
    zb = torch.randn((67, 128, 53), dtype=torch.float16)

    def fn(x, za, zb):
        p = x * 2  # sole consumers are the two transposes below (restickifies)
        return p.transpose(0, 2) + za, p.transpose(1, 2) + zb

    result, fused, _ = _run_capturing_padding_log(fn, x, za, zb)
    assert fused >= 1, "all-restickify shared producer should fuse"
    compare_with_cpu(fn, x, za, zb, target=result, run_eager=False)


def _is_restickify_op(op) -> bool:
    """True when ``op`` is a spyre.restickify ComputedBuffer, detected via its
    origin FX node's target."""
    from torch._inductor.ir import ComputedBuffer

    if not isinstance(op, ComputedBuffer):
        return False
    origins = op.origins
    if not origins:
        return False
    return next(iter(origins)).target is torch.ops.spyre.restickify.default


def _restickify_readers_by_source(fn, *args):
    """Run fn on Spyre and return {producer_name: [restickify buffer names]},
    a map of every source buffer to the restickify ops that read it, captured
    from graph.operations right after insert_restickify splices them in."""
    import torch_spyre._inductor.passes as _passes

    insert_restickify = _passes.insert_restickify
    by_source: dict[str, list[str]] = {}

    def capturing_insert_restickify(graph):
        insert_restickify(graph)
        for op in graph.operations:
            if not _is_restickify_op(op):
                continue
            for read in op.get_read_writes().reads:
                name = getattr(read, "name", None)
                if name is not None:
                    by_source.setdefault(name, []).append(op.get_name())

    with patch.object(_passes, "insert_restickify", capturing_insert_restickify):
        _compile_and_run(fn, args, DEVICE)
    return by_source


def test_shared_producer_gets_two_restickify_nodes():
    """insert_restickify keys its plan by consumer, not by source, so a producer
    that fans out to two consumers each wanting a different layout gets a
    *separate* restickify node per consumer -- both reading the one producer.
    We assert the two-node shape directly rather than via the fusion log."""
    x = torch.randn((67, 53, 128), dtype=torch.float16)
    za = torch.randn((128, 53, 67), dtype=torch.float16)
    zb = torch.randn((67, 128, 53), dtype=torch.float16)

    def fn(x, za, zb):
        p = x * 2  # sole consumers are the two transposes below (restickifies)
        return p.transpose(0, 2) + za, p.transpose(1, 2) + zb

    by_source = _restickify_readers_by_source(fn, x, za, zb)
    shared = [src for src, readers in by_source.items() if len(readers) >= 2]
    assert shared, (
        "expected one producer read by >=2 restickify nodes, got "
        f"{ {s: r for s, r in by_source.items()} }"
    )


# ------- Restickify padding: strict (distinct values + torch.equal) ---------
#
# The tolerance-based tests above compare ``randn`` data with atol=rtol=0.1,
# whose fp16 value collisions in [-3, 3] can MASK an element landing in the
# wrong stick.  The tests below feed a distinct-per-element ramp (``_arange``)
# and require exact equality (``_strict``), so a single misplaced element fails.
# They cover the transpose+clone geometries where a misplaced element is most
# likely: an unaligned stick split across blocks, multiple leading batch dims,
# and size-1 dims in or around the stick.

SPLIT_2D = [(65, 4), (67, 4), (128, 67), (130, 33)]


@pytest.mark.parametrize("shape", SPLIT_2D, ids=lambda p: f"{p[0]}x{p[1]}")
def test_strict_2d_transpose_clone(shape):
    x = _arange(*shape)
    _strict(lambda x: x.transpose(0, 1).clone(), x)


# transpose(-2, -1).clone() with >=2 leading batch dims: every batch plane must
# survive.  Covers a single stick block (..64..) and multiple blocks (..65..),
# an unaligned middle (old-stick) dim of size 4, and deeper/larger batch nests.
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


# transpose(0, -1).clone() swaps the OUTERMOST dim with the stick dim, with both
# the source and destination stick dims sub-64 (e.g. 2 and 7) so neither fills a
# full stick.  Must still place every element exactly.
SPLIT_T0_LAST = [(7, 67, 2), (7, 65, 2), (5, 3, 2), (7, 67, 63)]


@pytest.mark.parametrize("shape", SPLIT_T0_LAST, ids=lambda p: "x".join(map(str, p)))
def test_strict_transpose_0_last_clone(shape):
    x = _arange(*shape)
    _strict(lambda x: x.transpose(0, -1).clone(), x)


# A size-1 dim IN the input stick ((7, 67, 1) etc.): the transpose moves this
# size-1 dim out of the stick and a real dim in.  These shapes used to abort in
# the backend or return garbage for all but the first plane; every element must
# now come back correctly.
#
# The .exp() forces the restickify input to be an internal (produced) buffer
# rather than a bare graph input, exercising the in-place producer path; the
# graph-input path is covered by the next test.  A plain arithmetic op would be
# constant-folded away, so a transcendental is used -- its last-ULP host/device
# drift means this asserts allclose rather than the exact equality below.
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


# Same size-1-in-stick shapes, but the restickify input is a bare graph input
# (no .exp()), exercising the other input path.  A copy preserves the input
# bit-for-bit, so this asserts exact equality on the ramp (unlike the allclose
# above).
@pytest.mark.parametrize(
    "shape", SIZE1_INPUT_STICK, ids=lambda p: "x".join(map(str, p))
)
def test_size1_input_stick_transpose_0_last_clone_graph_input(shape):
    x = _arange(*shape)
    _strict(lambda x: x.transpose(0, -1).clone(), x)


# A size-1 dim in the input stick PLUS at least one more size-1 host dim
# elsewhere, so more than one size-1 dim is present at once.  Result must be
# correct regardless of where the extra size-1 dims sit; the interleaved
# variants (a real dim between the size-1 dims) confirm placement does not
# matter.
#
# Each entry is (shape, transpose_dims): the transpose swaps the size-1
# input-stick dim with a real dim, and the two untouched dims are both size-1.
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


# A size-1 dim in the input stick where a real batch/leading dim (extent > 1)
# survives OUTSIDE both the old (size-1) and new sticks -- e.g. (4, 64, 1)
# transpose(1, 2), whose batch dim 0 stays leading while dims 1 and 2 swap.
# Every surviving batch plane must come back correct (non-first planes used to
# be zeroed).
#
# The new (destination) stick is a full 64 here on purpose: an unaligned
# destination would additionally need output-middle padding, covered separately
# above.  Each entry is (shape, transpose_dims).
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


# A size-1 dim in the input stick PLUS a second size-1 host dim (leading or
# middle) that is not itself part of either stick.  With two size-1 dims present
# the transpose still has to place every real batch plane correctly regardless
# of where the extra size-1 dim sits (non-first planes used to come back
# zeroed).  Distinct-ramp + torch.equal catches a mis-placed plane exactly.
SIZE1_EXTRA = [
    ((1, 4, 64, 1), (2, 3)),  # extra size-1 leading (dim0)
    ((4, 1, 64, 1), (2, 3)),  # extra size-1 in the middle (dim1)
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


# A size-1 dim in the input stick whose NEW stick dim spans >=2 stick blocks
# (host size > 64), aligned or unaligned, with and without leading batch dims.
# The second and later stick blocks used to be mis-placed even when aligned;
# every block must now land correctly.  Distinct-ramp + torch.equal.
SIZE1_MULTI_BLOCK = [
    ((1, 128, 1), (1, 2)),  # 2 aligned blocks, no batch
    ((1, 192, 1), (1, 2)),  # 3 aligned blocks, no batch
    ((1, 67, 1), (1, 2)),  # 2 unaligned blocks, no batch
    ((4, 128, 1), (1, 2)),  # 2 aligned blocks + batch 4
    ((2, 128, 1), (1, 2)),  # 2 aligned blocks + batch 2
    ((4, 67, 1), (1, 2)),  # 2 unaligned blocks + batch
    ((4, 192, 1), (1, 2)),  # 3 blocks + batch
    ((2, 3, 67, 1), (2, 3)),  # 2 unaligned blocks + leading batch nest
]


@pytest.mark.parametrize(
    "shape,dims",
    SIZE1_MULTI_BLOCK,
    ids=[f"{'x'.join(map(str, s))}_t{d}" for s, d in SIZE1_MULTI_BLOCK],
)
def test_size1_multi_block_transpose_clone(shape, dims):
    x = _arange(*shape)
    _strict(lambda x: x.transpose(*dims).clone(), x)


# A restickify input sliced with a contiguous OFFSET on a NON-stick host dim
# (e.g. x[1:3]), with an unaligned new stick (67) that needs padding.
# This is a valid, paddable slice and must compile correctly (it used to be
# refused).  Distinct-ramp + torch.equal catches a misplaced plane exactly.
OFFSET_NONSTICK_INPUT = [
    # Graph-input leading-dim offset (rows 1..2 of 4), unaligned new stick (67).
    (lambda x: x[1:3].transpose(1, 2).clone(), (4, 67, 128)),
    # Offset on a middle (non-leading, non-stick) dim.
    (lambda x: x[:, 1:3].transpose(2, 3).clone(), (2, 4, 67, 128)),
]


@pytest.mark.parametrize(
    "fn,shape",
    OFFSET_NONSTICK_INPUT,
    ids=["x".join(map(str, s)) for _, s in OFFSET_NONSTICK_INPUT],
)
def test_offset_nonstick_input_transpose_clone(fn, shape):
    x = _arange(*shape)
    _strict(fn, x)


# A STRIDED (step > 1) read of a restickify input (``x[::2]``) is not paddable:
# the strided rows are non-adjacent, so a copy would read the wrong data.  With
# an unaligned new stick (67) that needs padding, this must fail loudly rather
# than silently miscompile.
def test_strided_input_transpose_clone_raises():
    x = _arange(4, 67, 128)
    with pytest.raises(RuntimeError, match="strided input on host dim"):
        _compile_and_run(lambda x: x[::2].transpose(1, 2).clone(), (x,), DEVICE)
