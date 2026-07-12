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

"""IR-level pass to pad y's K (row) dimension to a stick boundary for
BATCH_MATMUL_OP operations.  Runs in CustomPreSchedulingPasses immediately
after insert_restickify, when every ComputedBuffer has a FixedTiledLayout.

Only y is padded; x is left untouched.

For y, the following IR sequence is emitted:
  1. ComputedBuffer - output buffer allocation (FixedLayout)
  2. SpyreConstantFallback - fill constant (FixedLayout)
  3. ComputedBuffer - fill padding region (MutationLayoutSHOULDREMOVE)
  4. ComputedBuffer - copy input data (MutationLayoutSHOULDREMOVE)

y's padded buffer is built at the full K_padded host size by lower_pad_sequence.
reduction_ranges stays at K; the K→K_padded extension happens at SDSC codegen
time: _extend_matmul_k_to_padded in superdsc.py reads K_padded from y's
device_size and widens sdsc_iteration_space[K] to K_padded before
_create_sdsc_tensors runs.

x is left physically untouched.  The hardware masks within-stick elements of x
beyond the true K to zero, so extending the SDSC iteration to K_padded does not
introduce numerical error from x.

Deduplication of identical constants across multiple pad calls happens later
at the IR level via dedup_and_promote_constants.

x and y are identified via identify_matmul_inputs() using the BatchMatmul
generated_dim definition: y is the input whose index contains a symbol
present in the output but absent from x (N).  This handles M==K==N and
M=1 (decode phase) correctly.
"""

import math

import torch
from sympy import Expr
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    Buffer,
    ComputedBuffer,
    Operation,
    Pointwise,
    Reduction,
    TensorBox,
)
from torch._inductor.virtualized import V

from .constants import BATCH_MATMUL_OP
from .errors import Unsupported
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger
from .pass_utils import (
    concretize_expr,
    concretize_index,
    find_reduction_var,
    host_coordinates,
    identify_matmul_inputs,
    is_restickify,
    lower_pad_sequence,
    redirect_computed_buffer_reads,
    replace_computed_buffer_body,
    restickify_new_stick_pos,
)
from .views import compute_coordinates
from torch_spyre._C import SpyreTensorLayout, get_elem_in_stick

logger = get_inductor_logger("padding")


def compute_padding(cur_size: int, dtype: torch.dtype) -> int:
    stick_size = get_elem_in_stick(dtype)
    pad = (stick_size - (cur_size % stick_size)) % stick_size
    return pad


def _patch_env(graph_lowering) -> None:
    """Add view nodes (ReinterpretView) to env from name_to_users."""
    env: dict = {}
    for tbs in graph_lowering.name_to_users.values():
        for tb in tbs:
            if not tb.data.origins:
                continue
            tb_fx_node = list(tb.data.origins)[0]
            env[tb_fx_node] = tb
    graph_lowering.env.update(env)


def _move_ops_before(
    operations: list[Operation], new_ops: list[Operation], anchor: Operation
) -> None:
    """Relocate *new_ops* to sit immediately before *anchor* in *operations*.

    ``lower_pad_sequence`` appends new ops at the end of ``operations``; this
    helper moves them to just before the op that consumes them so topological
    order is preserved.
    """
    for o in new_ops:
        operations.remove(o)
    idx = operations.index(anchor)
    for i, o in enumerate(new_ops):
        operations.insert(idx + i, o)


def _find_arg_fx_node(arg_name: str) -> torch.fx.Node:
    """Return the FX node whose lowered TensorBox has the given buffer name.

    Buffer names are unique, but a single buffer can be reached through
    multiple FX nodes that present it at different sizes.  For example,
    mm_to_bmm_pass inserts an unsqueeze/reshape so the matmul inner_fn
    indexes x as 3D [1, M, K] even though the underlying buffer is 2D
    [M, K].  Both FX nodes lower to a TensorBox whose get_name() returns
    the same buffer name, but with different get_size() results.

    Returns the first candidate (the base buffer, with no view applied).
    Raises RuntimeError if no candidate exists.
    """
    graph_lowering = V.graph
    _patch_env(graph_lowering)
    candidates = [
        fx_node
        for fx_node, tb in graph_lowering.env.items()
        if isinstance(fx_node, torch.fx.Node)
        and isinstance(tb, TensorBox)
        and tb.get_name() == arg_name
    ]
    if not candidates:
        raise RuntimeError(f"no FX node found for buffer {arg_name!r}")
    return candidates[0]


def _rebuild_matmul(
    op: ComputedBuffer,
    y_padded_buf: Buffer,
    operations: list[Operation],
) -> ComputedBuffer:
    """Rebuild the matmul ComputedBuffer so y's loader reads from the padded buffer.

    Preserves the original inner_fn's x loading unchanged; only replaces y's
    loader with one that reads from the padded buffer.  reduction_ranges stays
    at K; the K→K_padded extension happens at SDSC codegen time via
    _extend_matmul_k_to_padded in superdsc.py.
    """
    reduction = op.data
    assert isinstance(reduction, Reduction)

    orig_inner_fn = reduction.inner_fn
    y_padded_loader = y_padded_buf.make_loader()
    y_ndim = len(y_padded_buf.get_size())
    y_batch_ndim = y_ndim - 2

    def new_inner_fn(
        index,
        reduction_index,
        _orig_inner_fn=orig_inner_fn,
        _y_loader=y_padded_loader,
        _y_batch_ndim=y_batch_ndim,
    ):
        # x_val comes from the original inner_fn; discard its y and replace below.
        x_val, _ = _orig_inner_fn(index, reduction_index)
        y_index = list(index[:_y_batch_ndim]) + list(reduction_index) + [index[-1]]
        y_val = _y_loader(y_index)
        return (x_val, y_val)

    object.__setattr__(reduction, "inner_fn", new_inner_fn)
    # reduction_ranges stays at K; no extension here.

    return replace_computed_buffer_body(op, reduction, operations)


def insert_bmm_padding(graph: GraphLowering) -> None:
    """
    Pad y's K (row) dimension for each BATCH_MATMUL_OP to a stick boundary.

    Mutates ``operations`` in place.  New buffers for y are inserted immediately
    before the matmul that consumes them to preserve topological order.

    x is left entirely untouched.  y's padded buffer is built at K_padded host
    size by lower_pad_sequence; reduction_ranges stays at K so the IR iteration
    space is unchanged.  The K→K_padded widening happens at SDSC codegen time.

    x and y are identified via identify_matmul_inputs() using the BatchMatmul
    generated_dim definition: y is the input whose index contains a symbol
    present in the output but absent from x (N).  This handles M==K==N and
    M=1 (decode phase) correctly.

    Deduplication of identical constants across multiple pad calls happens later
    at the IR level via dedup_and_promote_constants.
    """
    operations = graph.operations
    for op in list(operations):
        if not isinstance(op, ComputedBuffer):
            continue
        reduction = op.data
        if not isinstance(reduction, Reduction):
            continue
        if reduction.reduction_type != BATCH_MATMUL_OP:
            continue

        rw = op.get_read_writes()
        reads = [r for r in rw.reads if hasattr(r, "name")]
        if len(reads) != 2:  # noqa: PLR2004
            continue

        # Skip aligned-K matmuls early before any x/y identification.
        # Aligned-K matmuls need no padding regardless of input layout, and
        # skipping here avoids a spurious warning for e.g. decode-phase SDPA
        # attention where constant-folded dimensions cause identify_matmul_inputs
        # to fail.
        k_val = concretize_expr(reduction.reduction_ranges[0])
        first_buf = next(
            (graph.get_buffer(d.name) for d in reads if graph.get_buffer(d.name)),
            None,
        )
        assert first_buf is not None, (
            f"insert_bmm_padding: no input buffer found for matmul {op.get_name()}"
        )
        dtype = first_buf.get_dtype()
        if compute_padding(k_val, dtype) == 0:
            continue

        write_dep = next(iter(rw.writes))
        x_dep, y_dep = identify_matmul_inputs(reads, write_dep)
        if x_dep is None or y_dep is None:
            logger.warning(
                "insert_bmm_padding: could not identify x/y for %s, skipping",
                op.get_name(),
            )
            continue

        reduction_var = find_reduction_var(x_dep, write_dep)

        # y's K host dim: the dim whose host coordinate contains reduction_var.
        y_buf_tmp = graph.get_buffer(y_dep.name)
        y_host_k_dim: int | None = None
        if y_buf_tmp is not None and isinstance(
            y_buf_tmp.get_layout(), FixedTiledLayout
        ):
            y_h_coords = host_coordinates(y_buf_tmp.get_layout(), y_dep, None)
            y_host_k_dim = next(
                (
                    i
                    for i, c in enumerate(y_h_coords)
                    if reduction_var in c.free_symbols
                ),
                None,
            )

        x_name = x_dep.name
        y_name = y_dep.name
        x_buf = graph.get_buffer(x_name)
        y_buf = graph.get_buffer(y_name)
        if x_buf is None or y_buf is None:
            continue

        device = x_buf.get_device()
        pad = compute_padding(k_val, dtype)

        k_padded = k_val + pad

        logger.debug(
            "insert_bmm_padding: padding %s K=%d -> K=%d (pad=%d)",
            op.get_name(),
            k_val,
            k_padded,
            pad,
        )

        # The FX node for the matmul is used as the insertion anchor so padding
        # nodes are placed immediately before the matmul in the FX graph,
        # minimising their live range.
        matmul_fx_node = next(iter(op.origins))

        # --- Pad y only ---
        # y's K dimension is y's row (mb) dimension.  Padding it to K_padded
        # ensures rows K..K_padded-1 of y are zero-filled so the hardware
        # accumulates no contribution from those rows.
        # lower_pad_sequence builds the padded buffer at K_padded host size;
        # reduction_ranges is NOT changed.  superdsc._extend_matmul_k_to_padded
        # widens sdsc_iteration_space[K] to K_padded at SDSC codegen time,
        # reading K_padded from y's device_layout.device_size.
        y_size = [concretize_expr(s) for s in y_buf.get_size()]
        if y_host_k_dim is None:
            y_k_dim = len(y_size) - 2
        else:
            y_k_dim = y_host_k_dim
        y_padded_size = list(y_size)
        y_padded_size[y_k_dim] = k_padded
        y_fx_node = _find_arg_fx_node(y_name)

        y_orig_stl = y_buf.get_layout().device_layout
        y_padded_buf, y_new_ops = lower_pad_sequence(
            y_fx_node,
            padded_size=y_padded_size,
            device=device,
            dtype=dtype,
            dim=y_k_dim,
            insert_before=matmul_fx_node,
            orig_stl=y_orig_stl,
        )

        # --- Relocate new ops before the matmul ---
        # run_node appended them at the end of operations; move before op.
        _move_ops_before(operations, y_new_ops, op)

        # --- Rebuild matmul inner_fn to load y from the padded buffer ---
        # x is left entirely untouched: the original inner_fn's x loader is
        # preserved as-is.  Only y's loader is replaced with the padded buffer.
        _rebuild_matmul(op, y_padded_buf, operations)


# --------------------------------------------------------------------------- #
# insert_restickify_padding                                                   #
# --------------------------------------------------------------------------- #


def _single_free_sym(expr: Expr):
    """Return ``expr``'s sole free symbol, or None if it has zero or many."""
    syms = expr.free_symbols
    return next(iter(syms)) if len(syms) == 1 else None


def _device_coords(stl: SpyreTensorLayout, dep) -> list[Expr]:
    """Return device-space coordinate expressions for ``dep`` against ``stl``.

    Returns ``[]`` for a scalar / zero-dim layout (empty ``device_size``).
    """
    index = concretize_index(dep.index, set(dep.ranges.keys()))
    return compute_coordinates(stl.device_size, stl.stride_map, dep.ranges, index)


def _locate_size1_grow_dim(stl: SpyreTensorLayout, tiebreak=None) -> int | None:
    """Locate the size-1 (singleton) non-stick device dim that a restickify
    grows to a stick, or None if the choice is ambiguous.

    - a sole ``-1`` marker isolates the dim -> return it;
    - no ``-1`` marker -> fall back to a sole size-1 dim, else decline;
    - two-or-more ``-1`` markers -> hand the candidates to ``tiebreak`` (the
      output side breaks the tie by geometry; the input side passes None and
      declines).
    """
    device_size = [concretize_expr(s) for s in stl.device_size]
    stride_map = list(stl.stride_map)
    size1 = [d for d in range(len(device_size) - 1) if device_size[d] == 1]
    # A size-1 host dim carries no iteration symbol, so it can't be tracked
    # by symbol; it's instead the singleton device dim marked stride_map ==
    # -1 (the extent-1 marker set by coarse_tile._resize_device_layout).
    grow = [d for d in size1 if stride_map[d] == -1]
    if len(grow) == 1:
        return grow[0]
    if not grow:
        return size1[0] if len(size1) == 1 else None
    return tiebreak(grow) if tiebreak is not None else None


def _symbol_range_size(dep, sym) -> int | None:
    """Return the concretized size of loop symbol ``sym``'s range on ``dep``,
    or None if ``sym`` is None or absent from the dep's ranges.  Used to detect
    a narrowing slice: compare against the dim's declared size.
    """
    if sym is None or sym not in dep.ranges:
        return None
    return concretize_expr(dep.ranges[sym])


def _write_dep(op):
    """Return op's write dependency on the buffer it produces."""
    # Every ComputedBuffer here has exactly one named write; raise if not.
    return next(d for d in op.get_read_writes().writes if hasattr(d, "name"))


def _restickify_input(op, graph: GraphLowering):
    """Return ``(in_dep, in_buf, in_layout)`` for a restickify's single input, or
    ``(None, None, None)`` if ``op`` cannot be one (not a single-named-read
    pointwise copy whose input buffer has a FixedTiledLayout).
    """
    # Callers that already know op is a confirmed restickify can assume this
    # succeeds.
    reads = [r for r in op.get_read_writes().reads if hasattr(r, "name")]
    if len(reads) != 1:
        return None, None, None
    in_dep = reads[0]
    in_buf = graph.get_buffer(in_dep.name)
    if in_buf is None:
        return None, None, None
    in_layout = in_buf.get_layout()
    if not isinstance(in_layout, FixedTiledLayout):
        return None, None, None
    return in_dep, in_buf, in_layout


def _host_dim_carrying_sym(host_coords: list[Expr], sym) -> int | None:
    """Return the outermost host dim whose coordinate carries ``sym``, or None."""
    # A symbol whose range spans more than one tile can appear split across
    # several coordinates (e.g. v // 64 in one, v % 64 in another); the
    # outermost (lowest-index) one is the governing dim.
    for dim, coord in enumerate(host_coords):
        if sym in coord.free_symbols:
            return dim
    return None


def _stick_symbol(stl: SpyreTensorLayout, dep) -> object | None:
    """Return dep's within-stick device coordinate's free symbol against stl,
    or None if it has zero or many (e.g. a symbol-free size-1 stick, or a
    scalar / zero-dim layout with no coords at all).
    """
    device_coords = _device_coords(stl, dep)
    if not device_coords:
        return None
    # A coefficient or offset on the coordinate doesn't add free symbols, so
    # e.g. 2*(Mod(var, 32)) + 1 and Mod(var, 32) both resolve to {var}.
    return _single_free_sym(device_coords[-1])


def _identify_restickify(op: Operation, graph: GraphLowering):
    """Identify whether ``op`` is a restickify, and if so return its operands.

    A restickify is a single-input pointwise copy between two FixedTiledLayouts
    that lands a *different* host dim within the stick.  This is purely the
    "is it a restickify?" question; which dim to grow and which to leave
    alone is decided downstream by ``_pad_restickify_output`` /
    ``_pad_restickify_input``.

    Returns None if ``op`` is not a restickify, else ``(new_stick_dim,
    in_stick_dim)`` -- e.g. for a transpose that swaps which host dim sits in
    the stick:

    - ``new_stick_dim``: the input host dim that will occupy the *output's*
      stick (it is not yet in the input's stick).
    - ``in_stick_dim``: the input host dim currently in the *input's* stick
      (it becomes a plain non-stick device dim, the "old stick", on the
      output).

    Both are indices into the INPUT host dims (named for what they become on
    the output); neither is an output dim index, since a restickify re-tiles
    rather than preserving ranks.  The input dep, buffer, and layout are not
    returned: ``_restickify_input`` re-derives them from ``op``.
    """
    if not isinstance(op, ComputedBuffer):
        return None
    out_layout = op.get_layout()
    if not isinstance(out_layout, FixedTiledLayout):
        return None
    if not isinstance(op.data, Pointwise):
        return None

    in_dep, _in_buf, in_layout = _restickify_input(op, graph)
    if in_dep is None:
        return None

    in_coords = _device_coords(in_layout.device_layout, in_dep)
    out_coords = _device_coords(out_layout.device_layout, _write_dep(op))
    # A scalar / zero-dim layout has no coordinates, so it can't be a restickify.
    if not in_coords or not out_coords:
        return None
    # is_restickify is the single source of truth for this check, shared with
    # codegen's store side, so the two can't disagree on which ops restickify.
    if not is_restickify(in_coords, out_coords):
        return None

    def _input_host_dim_for_symbol(sym) -> int | None:
        """Return the input host dim carrying stick symbol ``sym`` (from either
        stick), or None.
        """
        in_host_coords = host_coordinates(in_layout, in_dep, None)
        if not in_host_coords:
            return None
        if sym is None:
            # A symbol-free stick means a size-1 host dim moved into stick
            # position.  Size-1 dims don't contribute to the device layout, so
            # any size-1 host dim yields the same device layout -- pick the
            # first; the physical dim to pad is re-derived from device-side
            # markers elsewhere (_restickify_input_device_dim /
            # _pad_restickify_output), not from this host index.
            ones = [i for i, s in enumerate(in_layout.size) if concretize_expr(s) == 1]
            return ones[0] if ones else None
        return _host_dim_carrying_sym(in_host_coords, sym)

    # in_stick_dim carries the input's own stick symbol; new_stick_dim carries
    # the OUTPUT's stick symbol (mirrors codegen: spyre_kernel.py locates it
    # the same way).  Both must resolve and differ, or the layout invariant
    # broke -- refuse loudly rather than skip and let codegen restickify an
    # unpadded buffer.
    in_stick_dim = _input_host_dim_for_symbol(
        _stick_symbol(in_layout.device_layout, in_dep)
    )
    new_stick_dim = _input_host_dim_for_symbol(
        _stick_symbol(out_layout.device_layout, _write_dep(op))
    )
    if in_stick_dim is None or new_stick_dim is None or new_stick_dim == in_stick_dim:
        raise Unsupported(
            "restickify padding: codegen restickifies but the pass could not "
            f"resolve its stick host dims for {op.get_name()} "
            "(unexpected: layout invariant broken)"
        )
    return new_stick_dim, in_stick_dim


def _device_dim_carrying_sym(stl: SpyreTensorLayout, write_dep, sym) -> int | None:
    """Return the outermost non-within-stick ``device_size`` dim whose device
    coordinate carries ``sym`` (the free symbol of some host dim), or None.
    """
    # Two host dims can share a dim size, so only the symbol unambiguously
    # identifies the dim.  A symbol whose range spans more than one tile can
    # appear split across several coordinates (e.g. v // 64 in one, v % 64 in
    # another); the outermost (lowest-index) one is the governing dim.
    device_coords = _device_coords(stl, write_dep)
    for dim in range(len(device_coords) - 1):
        if sym in device_coords[dim].free_symbols:
            return dim
    return None


def _pad_layout_device_dim(
    layout: FixedTiledLayout,
    device_dim: int,
    new_dim_size,
) -> FixedTiledLayout:
    """Return a copy of ``layout`` with one ``device_size`` dim grown to
    ``new_dim_size``, leaving the host size unchanged.
    """
    stl = layout.device_layout
    new_device_size = list(stl.device_size)
    new_device_size[device_dim] = new_dim_size
    # stride_map holds host strides, not host sizes, so it doesn't need to
    # change when only a device dim's size grows.
    padded_stl = SpyreTensorLayout(
        new_device_size, list(stl.stride_map), stl.device_dtype, stl.element_arrangement
    )
    host_size = [concretize_expr(s) for s in layout.size]
    host_stride = [concretize_expr(s) for s in layout.stride]
    return FixedTiledLayout(
        layout.device, layout.dtype, host_size, host_stride, padded_stl
    )


def _grow_size1_stick_dim(
    layout: FixedTiledLayout, device_dim: int
) -> FixedTiledLayout:
    """Return a copy of ``layout`` with a collapsed size-1 old-stick device dim
    grown to a full stick, giving it its real per-step host stride.

    Unlike ``_pad_layout_device_dim`` (which only grows the extent), the size-1
    collapse recorded the dim's host stride as the -1 singleton marker, so the
    grown dim must also be given a real stride: the product of the host extents
    (the size-1 host dim contributes 1).  This keeps DMA readback addressing the
    right plane once the dim iterates a full stick.
    """
    stl = layout.device_layout
    new_device_size = list(stl.device_size)
    new_device_size[device_dim] = stl.elems_per_stick()
    new_stride_map = list(stl.stride_map)
    new_stride_map[device_dim] = math.prod(concretize_expr(s) for s in layout.size)
    grown_stl = SpyreTensorLayout(
        new_device_size, new_stride_map, stl.device_dtype, stl.element_arrangement
    )
    host_size = [concretize_expr(s) for s in layout.size]
    host_stride = [concretize_expr(s) for s in layout.stride]
    return FixedTiledLayout(
        layout.device, layout.dtype, host_size, host_stride, grown_stl
    )


def _pad_restickify_output(
    op: Operation, graph: GraphLowering, in_stick_dim: int
) -> None:
    """Pad the output dim carrying the input's old stick to a stick boundary,
    so the second+ stick block and every batch plane land at the correct offset.

    Only the device layout grows; the tail rows are covered by
    ``_create_sdsc_tensors``'s backGap path and never read back. Skipped when
    the old stick's host coord has no surviving host dim, or its device dim is
    already stick-aligned.

    If the old stick collapsed to a size-1 device dim (no symbol), the same grow
    applies via ``_grow_size1_stick_dim``: the prealign restore
    (_restore_elided_restickify_stick_prealign) mints the stick's iteration
    symbol onto this grown dim, so growing it here keeps the allocation and the
    restored descriptor in agreement.
    """
    assert isinstance(op, ComputedBuffer)
    in_dep, _in_buf, in_layout = _restickify_input(op, graph)
    assert in_dep is not None  # op is a confirmed restickify
    in_host_coords = host_coordinates(in_layout, in_dep, None)
    old_sym = _single_free_sym(in_host_coords[in_stick_dim])
    out_layout = op.get_layout()
    write_dep = _write_dep(op)
    stl = out_layout.device_layout

    if old_sym is None:

        def _old_stick_size1_dim(grow: list[int]) -> int:
            # Two-or-more collapsed size-1 device dims: pick the demoted old stick to
            # grow to a full stick. Mirrors the restickify descriptor restore
            # (_restore_elided_restickify_stick_prealign): the restored old-stick
            # symbol lands at new_stick_pos -- the new stick's rank among the INPUT
            # device coords (a transpose swaps the two sticks' slots, so the old
            # stick lands where the new stick used to sit).  Growing that same slot
            # keeps allocation and descriptor in agreement.
            new_sym = _stick_symbol(stl, write_dep)
            if new_sym is not None:
                in_dev_coords = _device_coords(in_layout.device_layout, in_dep)
                new_stick_pos = restickify_new_stick_pos(in_dev_coords, {new_sym})
                if new_stick_pos is not None and new_stick_pos in grow:
                    return new_stick_pos
            # No single symbol (e.g. an all-ones batch): every candidate is a
            # zero-extent relabel yielding the same layout, so the first is safe.
            return grow[0]

        size1_dim = _locate_size1_grow_dim(stl, tiebreak=_old_stick_size1_dim)
        if size1_dim is not None:
            # Grow the collapsed old-stick dim to a full stick so the allocation
            # matches the restored descriptor's 64-plane write.  The prealign
            # restore (_restore_elided_restickify_stick_prealign) lands the
            # restored stick symbol on this same grown dim, so descriptor
            # coordinates and physical allocation agree.  The dim's host stride
            # was recorded as the -1 singleton marker (a size-1 host dim); its
            # real per-step stride is the product of the host extents.
            op.layout = _grow_size1_stick_dim(out_layout, size1_dim)
            logger.debug(
                "insert_restickify_padding: grew size-1 output %s device dim %d -> %d",
                op.get_name(),
                size1_dim,
                stl.elems_per_stick(),
            )
        return

    # Old stick collapsed to a size-1 output host dim (const-0 coord, no symbol):
    # nothing survives to misalign, so nothing to pad.
    out_host_coords = host_coordinates(out_layout, write_dep, None)
    if _host_dim_carrying_sym(out_host_coords, old_sym) is None:
        return
    device_dim = _device_dim_carrying_sym(stl, write_dep, old_sym)
    if device_dim is None:
        return
    # Already a stick multiple: stick blocks land aligned, no padding needed.
    old_dim_size = stl.device_size[device_dim]
    pad = compute_padding(old_dim_size, out_layout.dtype)
    if pad == 0:
        return

    new_dim_size = old_dim_size + pad
    op.layout = _pad_layout_device_dim(out_layout, device_dim, new_dim_size)
    logger.debug(
        "insert_restickify_padding: padded output %s device dim %d %d -> %d",
        op.get_name(),
        device_dim,
        old_dim_size,
        new_dim_size,
    )


def _restickify_input_device_dim(
    producer: ComputedBuffer, new_stick_dim: int
) -> int | None:
    """Return the producer device-size dim index that carries the input host
    dim ``new_stick_dim``, or None if the producer geometry does not expose it
    as a bumpable non-stick device dim.

    The caller bumps this dim to the stick boundary so the restickify's
    over-read lands inside the producer's own buffer instead of uninitialised
    HBM.
    """
    layout = producer.get_layout()
    write_dep = _write_dep(producer)
    host_coords = host_coordinates(layout, write_dep, None)
    stl = layout.device_layout
    sym = _single_free_sym(host_coords[new_stick_dim])
    if sym is not None:
        return _device_dim_carrying_sym(stl, write_dep, sym)
    # Size-1 host dim: locate the singleton producer device dim (declining if two
    # or more -1 markers make the choice ambiguous). Defensive backstop only --
    # _pad_restickify_input declines a size-1 new-stick read up front.
    return _locate_size1_grow_dim(stl)


def _grow_input_stick_dim(buf: ComputedBuffer, new_stick_dim: int, kind: str) -> None:
    """Grow ``buf``'s device_size on the dim carrying ``new_stick_dim`` up to the
    stick boundary, so the restickify's over-read lands inside ``buf``'s own
    allocation instead of uninitialised HBM.

    Shared by both input-padding entry points: the producer we own, and the
    identity clone of a graph input. Only device_size grows; host_size stays
    honest so a shared tracked named dim (e.g. a matmul's contraction dim)
    survives -- ``propagate_named_dims`` reads host_size as the dim's declared
    size.
    """
    device_dim = _restickify_input_device_dim(buf, new_stick_dim)
    assert device_dim is not None, (
        f"_grow_input_stick_dim: {kind} "
        f"{buf.get_name()} exposed no bumpable device dim for new stick dim "
        f"{new_stick_dim} -- a fresh clone always exposes one, and a producer "
        f"reaching here has a single non-size-1 free symbol on a restickified dim "
        f"that must project onto a non-within-stick device dim; a missing dim is a "
        f"can't-happen invariant violation."
    )

    layout = buf.get_layout()
    old_dim_size = layout.device_layout.device_size[device_dim]
    n = concretize_expr(layout.size[new_stick_dim])
    new_dim_size = n + compute_padding(n, layout.dtype)

    buf.layout = _pad_layout_device_dim(layout, device_dim, new_dim_size)

    logger.debug(
        "insert_restickify_padding: fused pad into %s %s device dim %d %d -> %d "
        "(new stick host dim %d)",
        kind,
        buf.get_name(),
        device_dim,
        old_dim_size,
        new_dim_size,
        new_stick_dim,
    )


def lower_identity_clone(
    arg_fx_node: torch.fx.Node,
    host_size: list[int],
    device: torch.device,
    dtype: torch.dtype,
    orig_stl: SpyreTensorLayout,
    insert_before: torch.fx.Node,
) -> tuple[ComputedBuffer, list[Operation]]:
    """Lower an identity ``aten.clone`` of ``arg_fx_node``, allocated at the
    ORIGINAL unpadded ``host_size``.

    The clone's host geometry is identical to the input, so its
    ``SpyreTensorLayout`` mirrors ``orig_stl`` verbatim; the caller bumps
    ``device_size`` on the stick-carrying dim afterwards (keeping this helper
    generic).

    Returns ``(clone_buf, new_ops)`` where ``clone_buf`` is the single new
    ComputedBuffer and ``new_ops`` is the (length-1) list of new IR operations.
    """
    graph_lowering = V.graph
    fx_graph = graph_lowering.graph

    ops_before = len(graph_lowering.operations)

    with fx_graph.inserting_before(insert_before):
        clone_fx = fx_graph.create_node(
            "call_function", torch.ops.aten.clone.default, args=(arg_fx_node,)
        )
        clone_fx.meta["val"] = torch.empty(host_size, dtype=dtype, device=device)

    clone_tb = graph_lowering.run_node(clone_fx)
    graph_lowering.env[clone_fx] = clone_tb

    # aten.clone lowers to an identity Pointwise; force realization so it becomes
    # a named ComputedBuffer rather than inlining into the consumer.
    clone_tb.data.realize()
    new_ops = graph_lowering.operations[ops_before:]
    assert len(new_ops) == 1 and isinstance(new_ops[0], ComputedBuffer), (
        f"lower_identity_clone: expected exactly one ComputedBuffer, got "
        f"{[type(o).__name__ for o in new_ops]}"
    )
    clone_buf = new_ops[0]

    host_layout = clone_buf.layout
    clone_stl = SpyreTensorLayout(
        list(orig_stl.device_size),
        list(orig_stl.stride_map),
        orig_stl.device_dtype,
        orig_stl.element_arrangement,
    )
    # insert_restickify_padding runs after propagate_spyre_tensor_layouts, so
    # run_node left this op with a FlexibleLayout; assign a FixedTiledLayout
    # here so the clone carries a real device layout.
    clone_buf.layout = FixedTiledLayout(
        host_layout.device,
        host_layout.dtype,
        host_layout.size,
        host_layout.stride,
        clone_stl,
    )

    assert clone_buf.origins, "lower_identity_clone: clone buffer has no origins"

    return clone_buf, new_ops


def _assert_input_paddable(
    op: ComputedBuffer, in_dep, in_layout, new_stick_dim: int
) -> None:
    """Raise ``Unsupported`` for restickify inputs the stick-boundary grow cannot
    pad yet, classifying each input dim's read by its coordinate.

    Not yet supported, and must fail loudly rather than miscompile:

    - **Strided** read of any dim (coord ``k*var``, k not in {0, 1}: step > 1 or
      reversed), e.g. ``x[::2].transpose(1, 2).clone()``.
    - **Narrowing slice on the new-stick dim** (iter range < dim size), e.g.
      ``x[:, :, 1:66, :].transpose(-2, -1).clone()``.

    TODO: both are liftable by double-restickifying -- a re-base copy that reads
    the sliced/strided source into a fresh stick-aligned buffer, then
    restickifies that (the grow can pad such a buffer). Once implemented,
    expressions that currently hit these raises would take that path instead
    and never reach this guard.

    Fine, and flow through unchanged:

    - **Contiguous offset** on a non-stick dim (coord ``var + c``), e.g.
      ``x[:, 1:, :]``.
    - **Broadcast** read (coeff 0), e.g.
      ``k.view(B, S, H, D).transpose(1, 2).transpose(2, 3)`` on a
      ``[B, S, H, 2, 1, D/2]`` input (see ``test_broadcast_input_transpose_clone``).
    """
    in_host_coords = host_coordinates(in_layout, in_dep, None)
    for i, coord in enumerate(in_host_coords):
        sym = _single_free_sym(coord)
        if sym is None:  # degenerate size-1 host dim, nothing to slice
            continue
        # Strided (k not in {0, 1}), on any dim: codegen carries only a contiguous
        # tail.  A broadcast (coeff 0) is read from the device layout, not this
        # coefficient, so it is not a stride and is left to flow through.
        if concretize_expr(coord.coeff(sym)) not in (0, 1):
            raise Unsupported(
                f"insert_restickify_padding: strided input on host dim "
                f"{i} of {op.get_name()} (coord {coord}) is not supported"
            )
        # A narrowing slice matters only on the new-stick dim (it starts the read
        # partway into a stick); a narrowed non-stick dim is a carried offset.
        if i != new_stick_dim:
            continue
        range_size = _symbol_range_size(in_dep, sym)
        dim_size = concretize_expr(in_layout.size[i])
        if range_size is not None and range_size != dim_size:
            raise Unsupported(
                f"insert_restickify_padding: sliced input on host dim "
                f"{i} of {op.get_name()} (iter range {range_size} != "
                f"dim size {dim_size}) is not supported"
            )


def _pad_restickify_input(
    op: Operation,
    graph: GraphLowering,
    new_stick_dim: int,
) -> None:
    """Read-side fix: ensure the restickify reads a grow-able ``ComputedBuffer``
    whose stick-carrying dim is padded to a stick boundary.

    Does nothing when there is no over-read to cover (the new-stick dim is
    already a stick multiple, or is a size-1 host dim). Otherwise validates the
    read is paddable (``_assert_input_paddable``), then grows the producer in
    place if we own it, or inserts and grows an identity clone for a graph
    input.
    """
    assert isinstance(op, ComputedBuffer)
    in_dep, in_buf, in_layout = _restickify_input(op, graph)
    assert in_dep is not None  # op is a confirmed restickify
    # Skip when the new-stick dim is already a stick multiple: the read never runs
    # past the true dim size. Keyed off the declared size, not the iteration range
    # (a slice landing here, e.g. x[3:66].transpose(0, 1), has range 63 < size 128
    # but must still skip -- the narrowing is a carried offset, not an unaligned
    # dim; test_sliced_transpose_stick_expr_compiles).
    host_size = [concretize_expr(s) for s in in_layout.size]
    if compute_padding(host_size[new_stick_dim], in_layout.dtype) == 0:
        return
    # Skip when the new-stick dim is size-1 (symbol-free coord): the read is fully
    # covered by the input's already-padded old stick, and the restickify restore
    # (_restore_elided_restickify_stick_prealign) supplies the padded lanes on the
    # elided output stick without ever reading them -- nothing here needs padding.
    in_host_coords = host_coordinates(in_layout, in_dep, None)
    if _single_free_sym(in_host_coords[new_stick_dim]) is None:
        return

    # Guard both branches: the offending read is the restickify's, not the buffer
    # we grow, so it defeats the producer-grow path just as it defeats the clone
    # path.
    _assert_input_paddable(op, in_dep, in_layout, new_stick_dim)

    # Producer arm: grow the ComputedBuffer we produced in place (cheap -- no
    # extra buffer or copy). Cannot fail here (see _grow_input_stick_dim's assert).
    if isinstance(in_buf, ComputedBuffer):
        _grow_input_stick_dim(in_buf, new_stick_dim, kind="producer")
        return

    # Clone arm (graph input only): materialise an identity clone ahead of the
    # restickify, grow it, and redirect the read to it.
    device = in_buf.get_device()
    if device is None:
        return

    dtype = in_layout.dtype

    in_fx = _find_arg_fx_node(in_dep.name)
    restickify_fx = next(iter(op.origins))
    clone_buf, new_ops = lower_identity_clone(
        in_fx,
        host_size=host_size,
        device=device,
        dtype=dtype,
        orig_stl=in_layout.device_layout,
        insert_before=restickify_fx,
    )

    # A clone we own always exposes the stick-carrying device dim; grow it
    # (the clone copies just the real rows).
    _grow_input_stick_dim(clone_buf, new_stick_dim, kind="clone")

    # Move the clone op to just before the restickify (run_node appends).
    _move_ops_before(graph.operations, new_ops, op)

    # Redirect the restickify to read the clone (wrap-not-reconstruct).
    redirect_computed_buffer_reads(
        op, {in_dep.name: clone_buf.get_name()}, graph.operations
    )


def insert_restickify_padding(graph: GraphLowering) -> None:
    """Pad a restickify's buffers so codegen's stick-boundary widening never
    touches uninitialized HBM.

    A restickify re-tiles a tensor so a different host dim lands within the
    stick.  Codegen widens both its read and its write to stick boundaries,
    which exposes two independent hazards, each with its own fix, attempted
    independently since either can apply without the other:

    - Write side (``_pad_restickify_output``): the output's old-stick host dim
      can land at the wrong physical offset.
    - Read side (``_pad_restickify_input``): the read can run past the true dim
      size into uninitialized HBM.

    Both fixes only grow a device dim size, never a host tensor dim size, so
    later passes that key off host sizes (e.g. ``propagate_named_dims``) are
    unaffected; the resulting host/device size gap is what codegen's backGap
    path fills in.
    """
    for op in list(graph.operations):
        match = _identify_restickify(op, graph)
        if match is None:
            continue
        new_stick_dim, in_stick_dim = match

        _pad_restickify_output(op, graph, in_stick_dim)
        _pad_restickify_input(op, graph, new_stick_dim)
