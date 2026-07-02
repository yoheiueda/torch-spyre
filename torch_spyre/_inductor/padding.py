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
from sympy import Integer
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    Buffer,
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
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
    is_stick_expr_offset_free,
    lower_pad_sequence,
    replace_computed_buffer_body,
)
from .views import compute_coordinates, matching_dim
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
        for new_op in y_new_ops:
            operations.remove(new_op)
        op_idx = operations.index(op)
        for i, new_op in enumerate(y_new_ops):
            operations.insert(op_idx + i, new_op)

        # --- Rebuild matmul inner_fn to load y from the padded buffer ---
        # x is left entirely untouched: the original inner_fn's x loader is
        # preserved as-is.  Only y's loader is replaced with the padded buffer.
        _rebuild_matmul(op, y_padded_buf, operations)


# --------------------------------------------------------------------------- #
# insert_restickify_padding                                                   #
# --------------------------------------------------------------------------- #


def _project_stick_host_dim(
    host_layout: FixedTiledLayout, stick_layout: FixedTiledLayout, dep
) -> int | None:
    """Return the host_layout host dim carrying stick_layout's within-stick
    coord under dep, or None if no unique canonical match exists.

    When host_layout is stick_layout this is the buffer's own within-stick
    host dim; when they differ, stick_layout's STL is projected through dep.
    Returns None unless the stick coord is valid.
    """
    host_coords = host_coordinates(host_layout, dep, None)
    stl = stick_layout.device_layout
    device_index = concretize_index(dep.index, set(dep.ranges.keys()))
    device_coords = compute_coordinates(
        stl.device_size, stl.stride_map, dep.ranges, device_index
    )
    if not host_coords or not device_coords:
        return None
    if not is_stick_expr_offset_free(device_coords[-1], stl.elems_per_stick()):
        return None
    return matching_dim(host_coords, device_coords[-1])


def _restickify_input_dep(op: Operation, graph: GraphLowering):
    """Return (in_dep, in_buf, in_layout, host_size, new_stick_dim,
    in_stick_dim) when op is a single-input pointwise copy whose output STL
    puts a different host dim within the stick than the input's does, else
    None.  ``new_stick_dim`` / ``in_stick_dim`` are the input host dims that
    become, respectively, the output's stick dim and the output's "old-stick"
    middle dim.

    Both stick dims are recovered in the input's host frame so they are
    directly comparable; the cross-buffer projection makes transpose work
    while reduce drops out as non-Pointwise and flatten drops out via the
    canonical-form filter in _project_stick_host_dim.  Sliced inputs are
    not filtered here — they raise Unsupported in the perm loop.
    """
    if not isinstance(op, ComputedBuffer):
        return None
    out_layout = op.get_layout()
    if not isinstance(out_layout, FixedTiledLayout):
        return None
    if not isinstance(op.data, Pointwise):
        return None

    rw = op.get_read_writes()
    reads = [r for r in rw.reads if hasattr(r, "name")]
    if len(reads) != 1:
        return None
    in_dep = reads[0]

    in_buf = graph.get_buffer(in_dep.name)
    if in_buf is None:
        return None
    in_layout = in_buf.get_layout()
    if not isinstance(in_layout, FixedTiledLayout):
        return None

    in_stick_dim = _project_stick_host_dim(in_layout, in_layout, in_dep)
    new_stick_dim = _project_stick_host_dim(in_layout, out_layout, in_dep)
    if in_stick_dim is None or new_stick_dim is None:
        return None
    if new_stick_dim == in_stick_dim:
        return None

    host_size = [concretize_expr(s) for s in in_layout.size]
    return in_dep, in_buf, in_layout, host_size, new_stick_dim, in_stick_dim


def _restickify_output_middle_device_dim(
    op: ComputedBuffer, in_dep, in_layout, in_stick_dim: int
) -> int | None:
    """Return the device-size entry index of the output's "old-stick" middle
    device dim, or None if it is stick-aligned and needs no padding.

    The output's middle dim is the host dim that carries the iter symbol of
    the input's old stick dim (``in_stick_dim``).  After the restickify it is a
    non-stick device entry whose true extent is the (small) old-stick host size.
    Padding is required whenever that device entry's extent is not a stick
    multiple, regardless of how many stick blocks the new stick dim spans or
    where the block axis sits relative to this entry: bumping the entry to a
    stick boundary widens the physical allocation so every batch plane and stick
    block lands at the correct offset.
    """
    out_layout = op.get_layout()
    odl = out_layout.device_layout
    stick_size = get_elem_in_stick(out_layout.dtype)

    wdep = next((d for d in op.get_read_writes().writes if hasattr(d, "name")), None)
    if wdep is None:
        return None

    in_host_coords = host_coordinates(in_layout, in_dep, None)
    old_syms = in_host_coords[in_stick_dim].free_symbols
    if len(old_syms) != 1:
        return None
    old_sym = next(iter(old_syms))

    out_host_coords = host_coordinates(out_layout, wdep, None)
    host_dim = next(
        (i for i, c in enumerate(out_host_coords) if old_sym in c.free_symbols),
        None,
    )
    if host_dim is None:
        return None

    didx = concretize_index(wdep.index, set(wdep.ranges.keys()))
    dcoords = compute_coordinates(odl.device_size, odl.stride_map, wdep.ranges, didx)

    device_entry = next(
        (k for k in range(len(dcoords) - 1) if old_sym in dcoords[k].free_symbols),
        None,
    )
    if device_entry is None:
        return None
    if odl.device_size[device_entry] % stick_size == 0:
        return None
    return device_entry


def _pad_restickify_output_middle(
    op: ComputedBuffer, in_dep, in_layout, in_stick_dim: int
) -> None:
    """Physically pad the restickify output's unaligned middle device dim to a
    stick boundary so the second+ stick block lands at the correct offset.

    See the caller for why this is needed.  Only the output buffer's device
    layout grows (middle device-size entry bumped to a stick multiple); the
    host size/stride stay logical, so the write index — and thus the logical
    output read by downstream consumers — are unchanged.  The over-allocated
    tail rows are written by ``_create_sdsc_tensors``'s existing
    ``dev_dim_size > it_dim_size`` backGap path and never read back.
    """
    device_entry = _restickify_output_middle_device_dim(
        op, in_dep, in_layout, in_stick_dim
    )
    if device_entry is None:
        return

    out_layout = op.get_layout()
    odl = out_layout.device_layout

    old_extent = odl.device_size[device_entry]
    new_extent = old_extent + compute_padding(old_extent, out_layout.dtype)

    # Bump ONLY the middle device-size entry; keep the stride_map untouched.
    #
    # The stride_map drives compute_coordinates, which derives the symbolic
    # device coordinates (including the within-stick ``Mod(var, 64)`` stick
    # expression) from the write index.  Rescaling it would desynchronise it
    # from the logical write index and yield an unsupported stick expression.
    # device_size, by contrast, is what superdsc's _calculate_device_stride
    # uses to compute the hardware SDSC strides, and what memory_planning uses
    # to size the allocation (prod of all but the within-stick entry).  So
    # bumping device_size alone widens the buffer to the stick-aligned geometry
    # and recomputes the block stride against the padded middle extent — which
    # is exactly what makes the second+ stick block land at the right physical
    # offset — while the stick expression stays clean.
    new_device_size = list(odl.device_size)
    new_device_size[device_entry] = new_extent
    padded_stl = SpyreTensorLayout(
        new_device_size,
        list(odl.stride_map),
        odl.device_dtype,
        odl.element_arrangement,
    )

    host_size = [concretize_expr(s) for s in out_layout.size]
    host_stride = [concretize_expr(s) for s in out_layout.stride]
    padded_layout = FixedTiledLayout(
        out_layout.device,
        out_layout.dtype,
        host_size,
        host_stride,
        padded_stl,
    )
    op.layout = padded_layout

    logger.debug(
        "insert_restickify_padding: padded output %s middle device dim %d %d -> %d",
        op.get_name(),
        device_entry,
        old_extent,
        new_extent,
    )


def _producer_stick_device_dim(
    producer: ComputedBuffer, new_stick_dim: int
) -> int | None:
    """Return the producer device-size entry index that carries the input host
    dim ``new_stick_dim`` (the dim the restickify turns into its new stick dim),
    or None if the producer geometry does not expose it as a bumpable middle
    device entry.

    The restickify over-reads this dim to the stick boundary; bumping the
    producer's device_size on the matching entry makes the producer allocate
    (and its backGap path leave defined) the widened tail, so the over-read
    lands inside the producer's own buffer instead of uninitialised HBM.

    The entry is located by the write index's host coordinate symbol (mirroring
    ``_restickify_output_middle_device_dim``), not by extent equality: two host
    dims can share an extent, and only the symbol match is unambiguous.  The
    within-stick entry is excluded.
    """
    layout = producer.get_layout()
    pdl = layout.device_layout

    wdep = next(
        (d for d in producer.get_read_writes().writes if hasattr(d, "name")), None
    )
    if wdep is None:
        return None

    host_coords = host_coordinates(layout, wdep, None)
    syms = host_coords[new_stick_dim].free_symbols
    if len(syms) != 1:
        return None
    sym = next(iter(syms))

    didx = concretize_index(wdep.index, set(wdep.ranges.keys()))
    dcoords = compute_coordinates(pdl.device_size, pdl.stride_map, wdep.ranges, didx)
    return next(
        (k for k in range(len(dcoords) - 1) if sym in dcoords[k].free_symbols),
        None,
    )


def _count_consumers(buf_name: str, graph: GraphLowering) -> int:
    """Number of operations that read ``buf_name``."""
    count = 0
    for op in graph.operations:
        try:
            reads = op.get_read_writes().reads
        except Exception:
            continue
        if any(getattr(r, "name", None) == buf_name for r in reads):
            count += 1
    return count


def _fuse_restickify_pad_into_producer(
    in_buf, new_stick_dim: int, n: int, pad: int, graph: GraphLowering
) -> bool:
    """Grow a single-consumer pointwise producer's output to the stick-aligned
    extent so the restickify reads its widened tail directly, avoiding the
    ``lower_pad_sequence`` copy (separate buffer + fill + copy + HBM round-trip).

    Mirrors ``_pad_restickify_output_middle``: only the producer's device_size
    on the entry carrying ``new_stick_dim`` grows (host size/stride and
    stride_map stay logical); the producer keeps computing its true ``n`` rows,
    and ``_create_sdsc_tensors``'s ``dev_dim_size > it_dim_size`` backGap path
    leaves the widened tail defined.  The restickify then over-reads the padded
    producer instead of uninitialised HBM (the over-read lands in the output's
    backGap-discarded band, so the tail value never reaches a read position).

    Only qualifies when the producer is an internal (non-output),
    single-consumer pointwise ``ComputedBuffer`` without a mutation layout;
    returns False otherwise so the caller falls back to ``lower_pad_sequence``.
    """
    if not isinstance(in_buf, ComputedBuffer):
        return False
    if not isinstance(in_buf.data, Pointwise):
        return False
    if isinstance(in_buf.layout, MutationLayoutSHOULDREMOVE):
        return False
    name = in_buf.get_name()
    if name in graph.get_output_names():
        return False
    if _count_consumers(name, graph) != 1:
        return False

    device_entry = _producer_stick_device_dim(in_buf, new_stick_dim)
    if device_entry is None:
        return False

    layout = in_buf.get_layout()
    pdl = layout.device_layout
    old_extent = pdl.device_size[device_entry]
    new_extent = n + pad

    # Bump ONLY the device_size entry; keep stride_map untouched (same rationale
    # as _pad_restickify_output_middle: rescaling stride_map would desynchronise
    # the within-stick expression from the logical write index).
    new_device_size = list(pdl.device_size)
    new_device_size[device_entry] = new_extent
    padded_stl = SpyreTensorLayout(
        new_device_size,
        list(pdl.stride_map),
        pdl.device_dtype,
        pdl.element_arrangement,
    )

    host_size = [concretize_expr(s) for s in layout.size]
    host_size[new_stick_dim] = new_extent
    host_stride = [concretize_expr(s) for s in layout.stride]
    in_buf.layout = FixedTiledLayout(
        layout.device,
        layout.dtype,
        host_size,
        host_stride,
        padded_stl,
    )

    logger.debug(
        "insert_restickify_padding: fused pad into producer %s device dim %d "
        "%d -> %d (host dim %d: %d -> %d)",
        name,
        device_entry,
        old_extent,
        new_extent,
        new_stick_dim,
        n,
        new_extent,
    )
    return True


def insert_restickify_padding(graph: GraphLowering) -> None:
    """Zero-pad a Restickify's input along the dim that becomes the new
    stick dim, when its extent is not a multiple of the stick size.

    Without padding, codegen widens the iteration space to a stick boundary
    and reads past the true extent — those tail elements come from
    uninitialized HBM and end up in the output, producing a value mismatch.

    Strategy: insert a stick-aligned, zero-filled copy of the input ahead of
    the Restickify (lower_pad_sequence) and rewrite the Restickify body to
    load from the padded buffer through a permuted index that maps each
    output iter dim to the corresponding input host dim.  The Restickify's
    ranges, layout, and device_layout are left untouched; codegen's existing
    stick-boundary widening reads from the zero-filled tail of the padded
    buffer instead of uninitialized HBM.
    """
    operations = graph.operations
    for op in list(operations):
        match = _restickify_input_dep(op, graph)
        if match is None:
            continue
        in_dep, in_buf, in_layout, host_size, new_stick_dim, in_stick_dim = match

        dtype = in_layout.dtype
        device = in_buf.get_device()
        if device is None:
            continue

        # --- Pad the output's unaligned "old-stick" middle dim (See #1756) ---
        # When the new stick dim spans more than one stick block, the output's
        # old-stick host dim becomes a non-stick MIDDLE device dim.  If its
        # extent is not a stick multiple, the restickify writes the second+
        # block at the wrong physical offset (the single-axis block stride only
        # lands correctly when the middle device dim is a stick multiple).  Pad
        # buf0's middle device dim up to a stick boundary, widening the
        # restickify's write into the (discarded) tail, and slice the logical
        # output back out.  This must run independently of input-stick padding:
        # e.g. a 128x67 transpose has an aligned input stick dim (128) but an
        # unaligned middle (67).
        _pad_restickify_output_middle(op, in_dep, in_layout, in_stick_dim)

        n = host_size[new_stick_dim]
        pad = compute_padding(n, dtype)
        if pad == 0:
            continue

        # Fuse the pad into a single-consumer pointwise producer when possible:
        # grow the producer's output to the stick-aligned extent (device_size
        # bump only) so the restickify over-reads its defined tail directly,
        # skipping the lower_pad_sequence copy.  Falls back below for graph
        # inputs, multi-consumer, non-pointwise, or mutation-layout producers.
        if _fuse_restickify_pad_into_producer(in_buf, new_stick_dim, n, pad, graph):
            continue

        padded_size = list(host_size)
        padded_size[new_stick_dim] = n + pad

        in_fx = _find_arg_fx_node(in_dep.name)
        restickify_fx = next(iter(op.origins))
        padded_buf, new_ops = lower_pad_sequence(
            in_fx,
            padded_size=padded_size,
            device=device,
            dtype=dtype,
            dim=new_stick_dim,
            insert_before=restickify_fx,
            orig_stl=in_layout.device_layout,
            fill_value=0.0,
        )

        # Move pad ops to just before the restickify (lower_pad_sequence appends).
        for o in new_ops:
            operations.remove(o)
        idx = operations.index(op)
        for i, o in enumerate(new_ops):
            operations.insert(idx + i, o)

        # Replace the restickify body with a Pointwise that reads padded_buf
        # through a permuted index, mapping each input host dim to the output
        # iteration position that drives it.  The output Pointwise iterates
        # op.data.ranges, so the inner_fn ``index`` list is positional over the
        # *output* host dims; output dim k carries the iter sym in
        # out_host_coords[k].  An input host dim whose coord carries the same sym
        # is loaded at index[k]; a degenerate (size-1) input dim carries no sym
        # and is loaded at constant 0.  Indexing by the output positions — rather
        # than the compressed input-sym order — is what makes size-1 host dims
        # (e.g. the leading-1 / seq=1 dims of a mid-stick torch.cat) work without
        # shifting the non-degenerate dims (See #1094).
        # op.data.ranges stays at the logical output extent; the stick-boundary
        # widening happens later in superdsc's _extend_restickify_to_padded
        # (Inductor's _simplify_loops would undo it if done here).
        in_host_coords = host_coordinates(in_layout, in_dep, None)
        write_dep = next(d for d in op.get_read_writes().writes if hasattr(d, "name"))
        out_host_coords = host_coordinates(op.get_layout(), write_dep, None)
        sym_to_out_pos = {
            next(iter(c.free_symbols)): k
            for k, c in enumerate(out_host_coords)
            if c.free_symbols
        }
        loader_perm: list[int | None] = []
        for i, coord in enumerate(in_host_coords):
            syms = coord.free_symbols
            if not syms:
                # Degenerate (size-1) input host dim: load at constant 0.
                loader_perm.append(None)
                continue
            assert len(syms) == 1, "_restickify_input_dep should have ensured this"
            sym = next(iter(syms))
            iter_extent = concretize_expr(in_dep.ranges[sym])
            dim_size = concretize_expr(in_layout.size[i])
            # Slice-detection lives here (not in the predicate): if the predicate
            # returned None for slices, codegen would silently produce wrong
            # output.  Raising here makes compilation fail loudly instead.
            # TODO: support sliced inputs (e.g. ``x[:, :, 1:66, :].transpose(-2, -1)``).
            if iter_extent != dim_size:
                raise Unsupported(
                    f"insert_restickify_padding: sliced input on host dim "
                    f"{i} of {op.get_name()} (iter range {iter_extent} != "
                    f"dim size {dim_size}) is not supported"
                )
            loader_perm.append(sym_to_out_pos[sym])
        old_pw = op.data
        padded_loader = padded_buf.make_loader()
        new_pw = Pointwise(
            device=old_pw.device,
            dtype=old_pw.dtype,
            inner_fn=lambda index, _loader=padded_loader, _perm=tuple(loader_perm): (
                _loader([Integer(0) if p is None else index[p] for p in _perm])
            ),
            ranges=old_pw.ranges,
        )
        replace_computed_buffer_body(op, new_pw, operations)
