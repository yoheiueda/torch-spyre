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

import torch
from sympy import Expr
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    Buffer,
    ComputedBuffer,
    Operation,
    Pointwise,
    Reduction,
    Scatter,
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


def _device_coords(stl: SpyreTensorLayout, dep) -> list[Expr]:
    """Return device-space coordinate expressions for ``dep`` against ``stl``."""
    index = concretize_index(dep.index, set(dep.ranges.keys()))
    return compute_coordinates(stl.device_size, stl.stride_map, dep.ranges, index)


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


def _stick_symbol(stl: SpyreTensorLayout, dep) -> object | None:
    """Return dep's within-stick device coordinate's free symbol against stl,
    or None if it has none (e.g. a symbol-free size-1 stick, or a scalar /
    zero-dim layout with no coords at all).
    """
    device_coords = _device_coords(stl, dep)
    if not device_coords:
        return None
    # A coefficient or offset on the coordinate doesn't add free symbols, so
    # e.g. 2*(Mod(var, 32)) + 1 and Mod(var, 32) both resolve to {var}.
    syms = device_coords[-1].free_symbols
    if not syms:
        return None
    assert len(syms) == 1, (
        f"_stick_symbol: within-stick coordinate {device_coords[-1]} carries "
        f"{len(syms)} free symbols, want exactly 1"
    )
    return next(iter(syms))


def _identify_restickify(op: Operation, graph: GraphLowering) -> bool:
    """Return whether ``op`` is a restickify.

    A restickify is a single-input pointwise copy between two FixedTiledLayouts
    that lands a *different* host dim within the stick.  This answers only the
    "is it a restickify?" question; each side then derives the stick dim it
    cares about from its own operand's stick symbol
    (``_pad_restickify_output`` from the input's old stick,
    ``_pad_restickify_input`` from the output's new stick).
    """
    if not isinstance(op, ComputedBuffer):
        return False
    out_layout = op.get_layout()
    if not isinstance(out_layout, FixedTiledLayout):
        return False
    if not isinstance(op.data, Pointwise):
        return False

    in_dep, _in_buf, in_layout = _restickify_input(op, graph)
    if in_dep is None:
        return False

    in_coords = _device_coords(in_layout.device_layout, in_dep)
    out_coords = _device_coords(out_layout.device_layout, _write_dep(op))
    # A scalar / zero-dim layout has no coordinates, so it can't be a restickify.
    if not in_coords or not out_coords:
        return False
    # is_restickify is the single source of truth for this check, shared with
    # codegen's store side, so the two can't disagree on which ops restickify.
    return is_restickify(in_coords, out_coords)


def _pad_layout_device_dim(
    layout: FixedTiledLayout,
    device_dim: int,
    new_dim_size,
) -> FixedTiledLayout:
    """Return a copy of ``layout`` with one ``device_size`` dim bumped to
    ``new_dim_size``, leaving the host size unchanged.
    """
    stl = layout.device_layout
    new_device_size = list(stl.device_size)
    new_device_size[device_dim] = new_dim_size
    # stride_map holds host strides, not host sizes, so it doesn't need to
    # change when only a device dim's size is bumped.
    padded_stl = SpyreTensorLayout(
        new_device_size, list(stl.stride_map), stl.device_dtype, stl.element_arrangement
    )
    host_size = [concretize_expr(s) for s in layout.size]
    host_stride = [concretize_expr(s) for s in layout.stride]
    return FixedTiledLayout(
        layout.device, layout.dtype, host_size, host_stride, padded_stl
    )


def _pad_restickify_output(op: Operation, graph: GraphLowering) -> None:
    """Pad the output dim carrying the input's old stick to a stick boundary,
    so the second+ stick block and every batch plane land at the correct offset.

    Only the device layout is bumped; the tail rows are covered by
    ``_create_sdsc_tensors``'s backGap path and never read back. Skipped when
    the old stick's host coord has no surviving host dim, or its device dim is
    already stick-aligned.

    When the old stick collapsed to a size-1 device dim (no iteration symbol),
    no output bump is needed: the prealign restore (_restickify_restore_elided_stick)
    synthesizes the stick's iteration symbol as an outermost dim, and align
    reconstructs a full 64-wide stick plane in the descriptor from the elided
    operand's floor/Mod decomposition -- so the allocation and descriptor
    already agree.
    """
    assert isinstance(op, ComputedBuffer)
    in_dep, _in_buf, in_layout = _restickify_input(op, graph)
    assert in_dep is not None  # op is a confirmed restickify
    # The old stick is the INPUT's own stick symbol (None when it collapsed to a
    # size-1 device dim).
    old_sym = _stick_symbol(in_layout.device_layout, in_dep)
    out_layout = op.get_layout()
    write_dep = _write_dep(op)
    stl = out_layout.device_layout

    if old_sym is None:
        # Old stick collapsed to a size-1 device dim (no iteration symbol): the
        # prealign restore mints and places the stick symbol, so nothing to bump.
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


def _pad_input_new_stick_dim(buf: ComputedBuffer, new_stick_dim: int) -> None:
    """Bump ``buf``'s device_size on the dim carrying ``new_stick_dim`` up to the
    stick boundary, so the restickify's over-read lands inside ``buf``'s own
    allocation instead of uninitialised HBM.

    Shared by both input-padding entry points: the producer we own, and the
    identity clone of a graph input. Only device_size is bumped; host_size
    stays honest so a shared tracked named dim (e.g. a matmul's contraction
    dim) survives -- ``propagate_named_dims`` reads host_size as the dim's
    declared size.

    ``new_stick_dim`` always carries a single live symbol here: a size-1
    new-stick dim is the OUTPUT-elided restickify, declined up front by
    ``_pad_restickify_input`` and restored by codegen instead, so it never
    reaches this bump path.

    The direct producer we own is a confirmed, paddable restickify input, so a
    missing single symbol or bumpable device dim is a broken invariant -- assert
    rather than skip.
    """
    layout = buf.get_layout()
    write_dep = _write_dep(buf)
    host_coords = host_coordinates(layout, write_dep, None)
    syms = host_coords[new_stick_dim].free_symbols
    assert len(syms) == 1, (
        f"_pad_input_new_stick_dim: new-stick dim {new_stick_dim} on "
        f"{buf.get_name()} carries {len(syms)} free symbols, want exactly 1"
    )
    device_dim = _device_dim_carrying_sym(
        layout.device_layout, write_dep, next(iter(syms))
    )
    assert device_dim is not None, (
        f"_pad_input_new_stick_dim: {buf.get_name()} exposed no bumpable device dim "
        f"for new stick dim {new_stick_dim}"
    )
    # Bump device_size[device_dim] up to the stick boundary of the host size at
    # new_stick_dim; a no-op when the dim is already a stick multiple.
    n = concretize_expr(layout.size[new_stick_dim])
    new_dim_size = n + compute_padding(n, layout.dtype)
    if concretize_expr(layout.device_layout.device_size[device_dim]) != new_dim_size:
        buf.layout = _pad_layout_device_dim(layout, device_dim, new_dim_size)


def _recover_restickify_transpose_perm(
    read_dep: MemoryDep, prod_write_dep: MemoryDep, ranges: list[int]
) -> list[int] | None:
    """Recover which producer storage dim each restickify iteration dim reads,
    by comparing the restickify's read against the producer's own write.

    Returns ``perm`` with ``perm[i]`` = the producer storage dim that restickify
    iteration dim ``i`` reads (identity for a plain, non-transposing restickify).
    Returns ``None`` when a read coefficient has no matching producer stride (an
    unexpected read shape), so the caller can decline to rewrite.

    read_dep (the restickify's read) and prod_write_dep (the producer's write)
    index the same buffer, so their affine indices carry the SAME set of stride
    coefficients; only which iteration var each coefficient multiplies differs, and
    that difference IS the transpose.
    """
    # Map each producer stride coefficient to the storage dim it addresses...
    prod_index = prod_write_dep.index.expand()
    coeff_to_prod_dim = {
        prod_index.coeff(v): d for d, v in enumerate(prod_write_dep.var_names)
    }
    read_index = read_dep.index.expand()
    read_vars = list(read_dep.var_names)
    perm: list[int | None] = [None] * len(ranges)
    used: set[int] = set()
    var_i = 0
    # ...then walk the restickify's iteration ranges, looking up the coefficient on
    # each read var to find the producer dim that iteration dim reads. ranges drives
    # the walk (not read_vars) because a size-1 iteration dim carries no symbol and
    # so is absent from read_dep's vars.
    for i, extent in enumerate(ranges):
        if extent == 1:
            continue  # size-1 dim: no symbol in the dep, filled from leftovers below
        coeff = read_index.coeff(read_vars[var_i])
        var_i += 1
        prod_dim = coeff_to_prod_dim.get(coeff)
        if prod_dim is None:
            return None
        perm[i] = prod_dim
        used.add(prod_dim)
    # Assign the size-1 dims skipped above from the leftover (unmatched) producer dims.
    leftover = iter(d for d in range(len(ranges)) if d not in used)
    return [next(leftover) if p is None else p for p in perm]


def _make_restickify_scatter(data: Pointwise, perm: list[int]) -> Scatter:
    """Build a ``Scatter`` equivalent to the transposing restickify ``data`` but
    iterating in the producer's storage order rather than the transposed order.

    This carries the transpose on the store instead of the read, so the whole
    fused chain iterates one order. The ``Scatter`` reads the producer straight and
    permutes only where it writes into the fresh output buffer.

    ``perm[i]`` = the producer storage dim read by original iteration dim ``i``
    (from ``_recover_restickify_transpose_perm``).
    """
    n = len(perm)
    # inv is perm's inverse: it maps a producer dim back to the original
    # iteration dim over it.
    inv = [0] * n
    for i, prod_dim in enumerate(perm):
        inv[prod_dim] = i
    old_ranges = list(data.ranges)
    # new_ranges reorders the original ranges into producer order via inv:
    # producer dim d iterates the extent of original dim inv[d].
    new_ranges = [old_ranges[inv[d]] for d in range(n)]
    orig_inner_fn = data.inner_fn

    # index is producer-order. The original loader was built to be indexed in the
    # transposed (original iteration) order, so rebuild that index: original dim i
    # takes the coordinate at producer dim perm[i].
    def inner_fn(index):
        return orig_inner_fn([index[perm[i]] for i in range(n)])

    # Apply the same remap to place each producer-order iteration at its
    # transposed slot in the output storage.
    def output_indexer(index):
        return [index[perm[i]] for i in range(n)]

    return Scatter(
        device=data.device,
        dtype=data.dtype,
        inner_fn=inner_fn,
        ranges=new_ranges,
        output_indexer=output_indexer,
    )


def _rewrite_restickify_to_scatter(op: ComputedBuffer, in_buf: ComputedBuffer) -> None:
    """Express a transposing restickify ``op`` as a ``Scatter`` so its fused
    kernel reads its producer in the producer's own order.

    ``op`` is a restickify (a ``.contiguous()`` copy) whose input and output
    layouts differ by a transpose, and ``in_buf`` is its producer: the op fused
    immediately upstream, whose output buffer ``op`` reads. This backend fuses ``op``,
    ``in_buf``, and any further pointwise ops around them into one kernel -- a single
    loop nest (one SDSC bundle) over one iteration space, with every op indexed by the
    same loop variables. That one shared iteration order is a requirement of the fused
    form: every op in the chain must be expressible over the one set of loop variables.

    The requirement is not intrinsic to the hardware; it comes from how this backend
    fuses. Upstream Inductor fusion is disabled (the Spyre can_fuse heuristics all
    return False); ``spyre_fuse_nodes`` instead groups contiguous Spyre ops into one
    FusedSchedulerNode compiled to one SDSC bundle, reusing Inductor's fused-node and
    ops-handler framework -- which calls every op's loader and store with the same
    iteration vars. So a fused bundle cannot reindex between ops; a transpose can only
    ride a load or a store.

    A transpose is a view, so it carries no op of its own; it survives only as the
    order in which ``op`` reads ``in_buf``. As a plain copy ``op`` iterates the output
    (transposed) order and reads a transposed view of ``in_buf``, while ``in_buf`` was
    written in its own order -- so within that single loop the interior buffer is read
    in a different order than it was written, and lanes land permuted. Rewriting ``op``
    to read the producer straight and carry the transpose on its store instead
    (``_make_restickify_scatter``) keeps the whole chain on one iteration order, with
    no interior buffer read in an order other than it was written -- the store into the
    fresh output buffer is the one place a reorder is safe, since it is not an interior
    fused buffer.

    Example -- ``(x + x).mul(2.0).transpose(0, -1).contiguous()`` on ``[2, 3, 4]``
    lowers to a sequence of buffers::

        buf0 = add(x, x)          # Pointwise
        buf1 = mul(buf0, 2.0)     # Pointwise, reads buf0
        buf2 = restickify(buf1)   # the .contiguous() after a transpose view

    The transpose survives only as the order in which ``buf2`` reads ``buf1``. Fusion
    merges ``buf0``, ``buf1``, ``buf2`` into one kernel with one loop nest (one SDSC
    bundle): for each iteration index the loop body computes ``add``, feeds it to
    ``mul``, and feeds that to the restickify store, all indexed by the same loop
    variables. As a plain copy the restickify iterates the output ``[4, 3, 2]`` order
    and reads ``buf1`` transposed, while ``mul`` wrote ``buf1`` in ``[2, 3, 4]`` order
    -- so ``buf1`` is the transpose-crossing buffer, read differently than written. As
    a Scatter the restickify reads ``buf1`` straight in ``[2, 3, 4]`` order and applies
    the transpose only when storing into ``buf2``, so ``add``, ``mul``, and the read
    all share the ``[2, 3, 4]`` order.

    Recovers the transpose from the read vs. the producer's write
    (``_recover_restickify_transpose_perm``) and rebuilds ``op.data``. Declines to a
    no-op, leaving the read-side bump as the only fix, when the producer is not a
    Pointwise buffer or the read carries no transpose. Raises ``Unsupported`` when the
    transpose is unrecoverable (an unexpected read shape), since leaving such a read as
    a Pointwise could silently permute lanes. (Other properties this relies on --
    op.data Pointwise, a tiled producer layout, a single indexed read of in_buf -- are
    already guaranteed by restickify identification and asserted below.)
    """
    data = op.data
    assert isinstance(data, Pointwise), "a restickify's op.data is always Pointwise"
    # The rewrite recovers the transpose by matching the restickify's read against the
    # producer's affine write index (_recover_restickify_transpose_perm), which
    # requires a producer that writes such an index -- a Pointwise ComputedBuffer with
    # a tiled layout. Other producers (e.g. a Reduction, whose in_buf.data is a
    # Reduction, a Loops sibling of Pointwise, not an instance of it) fall through to
    # the read-side bump alone; the rewrite is not attempted for them.
    #
    # TODO: a non-Pointwise producer could be given the rewrite by cloning it into a
    # Pointwise buffer (lower_identity_clone, as the graph-input arm does) and reading
    # that. Not done here: the clone would likely re-fuse into the same bundle
    # (spyre_fuse_nodes groups contiguous Spyre nodes regardless of realization), so
    # whether it yields the interior boundary the rewrite needs is unvalidated.
    if not isinstance(in_buf.data, Pointwise):
        return
    # _restickify_input already required a FixedTiledLayout producer to confirm op is
    # a restickify, so this is an invariant here.
    assert isinstance(in_buf.get_layout(), FixedTiledLayout)
    # op has exactly one indexed read, of in_buf: _restickify_input required a single
    # name-bearing read (in_dep, from which in_buf was fetched), and
    # _identify_restickify then took its device coordinates via dep.index -- which a
    # StarDep does not have -- so in_dep is a MemoryDep of in_buf. Hence this is an
    # invariant here, not a case to decline.
    reads = [d for d in op.get_read_writes().reads if isinstance(d, MemoryDep)]
    assert len(reads) == 1 and reads[0].name == in_buf.get_name()
    ranges = [concretize_expr(r) for r in data.ranges]
    perm = _recover_restickify_transpose_perm(reads[0], _write_dep(in_buf), ranges)
    # A None perm means a read coefficient had no matching producer stride, so the
    # read is a shape the recovery does not model. We cannot tell whether it hides a
    # real transpose -- if it does, leaving op as a Pointwise would fuse a
    # differently-ordered interior read into one kernel and silently permute lanes,
    # the very miscompile this rewrite prevents. So fail loudly rather than risk it,
    # matching _assert_input_paddable's treatment of reads it cannot pad.
    if perm is None:
        raise Unsupported(
            f"insert_restickify_padding: could not recover the transpose of "
            f"restickify {op.get_name()} from its read of {in_buf.get_name()} "
            f"(unexpected read shape); refusing to risk a permuted-lane miscompile"
        )
    # An identity perm means op reads its producer in producer order, so there is no
    # transpose to carry on the store. This covers a plain, non-transposing restickify
    # (e.g. an insert_restickify retiling the same host layout), AND a restickify whose
    # transpose was already absorbed upstream in the fused chain -- a pointwise op
    # sitting between the transpose and this .contiguous() (e.g.
    # ``x.transpose(0, -1).mul(3.0).contiguous()``) is the crossing edge, read in the
    # transposed order it was written, so this restickify reads that op straight.
    # Either way there is no crossing buffer here, so op stays on its Pointwise codegen
    # path with the read-side bump as the only fix.
    if perm == list(range(len(perm))):
        return
    op.data = _make_restickify_scatter(data, perm)
    logger.debug(
        "insert_restickify_padding: rewrote restickify %s to Scatter (transpose "
        "perm %s on producer %s)",
        op.get_name(),
        perm,
        in_buf.get_name(),
    )


def _assert_input_paddable(
    op: ComputedBuffer, in_dep, in_layout, new_stick_dim: int
) -> None:
    """Raise ``Unsupported`` for restickify inputs the stick-boundary bump cannot
    pad yet, classifying each input dim's read by its coordinate.

    Not yet supported, and must fail loudly rather than miscompile:

    - **Strided** read of any dim (coord ``k*var``, k not in {0, 1}: step > 1 or
      reversed), e.g. ``x[::2].transpose(1, 2).clone()``.
    - **Narrowing slice on the new-stick dim** (iter range < dim size), e.g.
      ``x[:, :, 1:66, :].transpose(-2, -1).clone()``.

    TODO: both are liftable by double-restickifying -- a re-base copy that reads
    the sliced/strided source into a fresh stick-aligned buffer, then
    restickifies that (the bump can pad such a buffer). Once implemented,
    expressions that currently hit these raises would take that path instead
    and never reach this guard.
    """
    in_host_coords = host_coordinates(in_layout, in_dep, None)
    for i, coord in enumerate(in_host_coords):
        syms = coord.free_symbols
        if not syms:  # degenerate size-1 host dim, nothing to slice
            continue
        assert len(syms) == 1, (
            f"insert_restickify_padding: host dim {i} of {op.get_name()} "
            f"(coord {coord}) carries multiple free symbols -- an interleaved "
            f"index this pass's per-dim strided/sliced classification cannot "
            f"read; a restickify input must not reach this shape"
        )
        sym = next(iter(syms))
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
        range_size = (
            concretize_expr(in_dep.ranges[sym]) if sym in in_dep.ranges else None
        )
        dim_size = concretize_expr(in_layout.size[i])
        if range_size is not None and range_size != dim_size:
            raise Unsupported(
                f"insert_restickify_padding: sliced input on host dim "
                f"{i} of {op.get_name()} (iter range {range_size} != "
                f"dim size {dim_size}) is not supported"
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


def _pad_restickify_input(op: Operation, graph: GraphLowering) -> None:
    """Read-side fix: ensure the restickify reads a bump-able ``ComputedBuffer``
    whose stick-carrying dim is padded to a stick boundary.

    Does nothing when there is no over-read to cover (the new-stick dim is a
    size-1 host dim, or is already a stick multiple). Otherwise validates the
    read is paddable (``_assert_input_paddable``), then bumps the producer in
    place if we own it, or inserts and bumps an identity clone for a graph
    input.
    """
    assert isinstance(op, ComputedBuffer)
    in_dep, in_buf, in_layout = _restickify_input(op, graph)
    assert in_dep is not None  # op is a confirmed restickify
    # A None new-stick symbol means it's a size-1 host dim: codegen's restore
    # (_restickify_restore_elided_stick) covers the elided output stick without reading
    # the input, so there's nothing here to pad.
    out_stick_sym = _stick_symbol(op.get_layout().device_layout, _write_dep(op))
    if out_stick_sym is None:
        return
    # A resolving symbol maps to exactly one input host dim; a non-resolving one
    # is a broken layout invariant, not an unsupported input -- assert.
    in_host_coords = host_coordinates(in_layout, in_dep, None)
    new_stick_dim = _host_dim_carrying_sym(in_host_coords, out_stick_sym)
    assert new_stick_dim is not None, (
        f"restickify padding: no input host dim carries new-stick symbol "
        f"{out_stick_sym} for {op.get_name()} (layout invariant broken)"
    )
    # Skip if already a stick multiple. Keyed off declared size, not iteration
    # range -- a slice here (e.g. x[3:66].transpose(0, 1)) narrows the range but
    # must still skip, since the narrowing is a carried offset, not misalignment.
    host_size = [concretize_expr(s) for s in in_layout.size]
    if compute_padding(host_size[new_stick_dim], in_layout.dtype) == 0:
        return

    # Applies to both arms below: an unpaddable read defeats bumping the
    # producer just as much as bumping a clone.
    _assert_input_paddable(op, in_dep, in_layout, new_stick_dim)

    def log_bumped(kind: str, buf: ComputedBuffer) -> None:
        logger.debug(
            "insert_restickify_padding: bumped %s %s to device_size %s "
            "(new stick host dim %d)",
            kind,
            buf.get_name(),
            buf.get_layout().device_layout.device_size,
            new_stick_dim,
        )

    # The input is a preceding op's output buffer, not a graph input, so we can
    # bump it in place (cheap -- no extra buffer or copy).
    if isinstance(in_buf, ComputedBuffer):
        _pad_input_new_stick_dim(in_buf, new_stick_dim)
        log_bumped("producer", in_buf)
        # The bump only keeps the over-read in bounds. If this restickify also
        # transposes, its fused kernel still reads an interior buffer in a
        # different order than it was written; expressing it as a Scatter that
        # reads the producer straight keeps the kernel on one iteration order (a
        # no-op for a non-transposing restickify).
        _rewrite_restickify_to_scatter(op, in_buf)
        return

    # The input is a graph input: materialise, bump, and redirect the read to
    # an identity clone inserted ahead of the restickify.
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

    # The clone mirrors the input's layout, so the same dim can be bumped.
    _pad_input_new_stick_dim(clone_buf, new_stick_dim)
    log_bumped("clone", clone_buf)

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
      size into uninitialized HBM.  A transposing restickify additionally has its
      producer read in the original order and its store permuted
      (``_rewrite_restickify_to_scatter``), so no interior buffer of a fused chain
      is written in one order and read in another.

    The padding fixes only bump a device dim size, never a host tensor dim size,
    so later passes that key off host sizes (e.g. ``propagate_named_dims``) are
    unaffected; the resulting host/device size gap is what codegen's backGap
    path fills in.  The Scatter rewrite likewise leaves host sizes untouched: it
    reindexes iteration and the store, not the declared shape.

    Neither fix bumps a size-1 (elided) stick dim: with no iteration symbol,
    upstream Inductor gives it no coordinate to bump here, and doing so before
    align_tensors runs produces a fractional coordinate ``normalize_coordinates``
    rejects (see ``_restickify_restore_elided_stick`` in spyre_kernel.py, which
    bumps it later, right before align).
    """
    for op in list(graph.operations):
        if _identify_restickify(op, graph):
            _pad_restickify_output(op, graph)
            _pad_restickify_input(op, graph)
