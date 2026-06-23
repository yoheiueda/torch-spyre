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
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    Buffer,
    ComputedBuffer,
    Operation,
    Pointwise,
    Reduction,
    ReinterpretView,
    TensorBox,
)
from torch._inductor.virtualized import V

from .constants import BATCH_MATMUL_OP
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger
from .pass_utils import (
    concretize_expr,
    device_coordinates,
    find_reduction_var,
    identify_matmul_inputs,
    host_coordinates,
    lower_pad_preserve_layout,
    lower_pad_sequence,
    replace_computed_buffer_body,
)
from .views import matching_dim
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
            y_h_coords = host_coordinates(y_buf_tmp.get_layout(), y_dep)
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


def _restickify_input_dep(op: Operation, graph: GraphLowering):
    """Return (in_dep, in_buf, in_layout) when ``op`` is a ComputedBuffer
    whose layout swaps the stick dimension relative to its single input.
    Returns None otherwise.

    A "restickify" here is identified structurally: a single-input pointwise
    copy whose input's within-stick host coord uses a different symbol than
    the output's within-stick device coord.  This catches both the explicit
    ``torch.ops.spyre.restickify`` FX op (inserted by ``insert_restickify``)
    and the fused permute+clone ComputedBuffer that lowers to ReStickifyOpHBM
    directly.
    """
    if not isinstance(op, ComputedBuffer):
        return None
    out_layout = op.get_layout()
    if not isinstance(out_layout, FixedTiledLayout):
        return None
    from torch._inductor.ir import Pointwise as _Pointwise

    if not isinstance(op.data, _Pointwise):
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

    in_host_coords = host_coordinates(in_layout, in_dep)
    out_dev_coords = device_coordinates(out_layout.device_layout, in_dep)
    if in_host_coords[-1].free_symbols == out_dev_coords[-1].free_symbols:
        return None
    return in_dep, in_buf, in_layout


def insert_restickify_padding(graph: GraphLowering) -> None:
    """Zero-pad a Restickify's input buffer when the dim that becomes the new
    stick dim is not a multiple of the stick size.

    Without padding, the codegen widens the SDSC iteration space to a stick
    boundary (#2112) and reads ceil(N/stick) sticks from the input — the tail
    of the last stick contains uninitialized HBM, which the Restickify writes
    into the output's stick layout, producing the mismatch tracked by #1756.

    Strategy: insert a stick-aligned, zero-filled copy of the input ahead of
    the Restickify (lower_pad_sequence) and rewrite the Restickify body to
    load from the padded buffer.  The Restickify's ranges, layout, and
    device_layout are NOT touched — they already describe the padded
    iteration correctly via #2112's ceil-div + backGap path.
    """
    operations = graph.operations
    for op in list(operations):
        match = _restickify_input_dep(op, graph)
        if match is None:
            continue
        in_dep, in_buf, in_layout = match

        out_layout = op.get_layout()
        in_host_coords = host_coordinates(in_layout, in_dep)
        out_dev_coords = device_coordinates(out_layout.device_layout, in_dep)
        # The dim becoming a stick on the output, expressed in input host axes.
        new_stick_dim = matching_dim(in_host_coords, out_dev_coords[-1])
        if new_stick_dim is None:
            continue

        host_size = [concretize_expr(s) for s in in_layout.size]
        dtype = in_layout.dtype
        n = host_size[new_stick_dim]
        pad = compute_padding(n, dtype)
        if pad == 0:
            continue

        device = in_buf.get_device()
        if device is None:
            continue

        padded_size = list(host_size)
        padded_size[new_stick_dim] = n + pad

        in_fx = _find_arg_fx_node(in_dep.name)
        restick_fx = next(iter(op.origins))
        padded_buf, new_ops = lower_pad_preserve_layout(
            in_fx,
            padded_size=padded_size,
            device=device,
            dtype=dtype,
            dim=new_stick_dim,
            insert_before=restick_fx,
            orig_stl=in_layout.device_layout,
            fill_value=0.0,
        )

        # Move pad ops to just before the restickify (lower_pad_sequence appends).
        for o in new_ops:
            operations.remove(o)
        idx = operations.index(op)
        for i, o in enumerate(new_ops):
            operations.insert(idx + i, o)

        # Rewire: replace the restickify body with a Pointwise whose loader
        # reads ``padded_buf`` through a ReinterpretView that mirrors the
        # original input view's stride pattern.  This is the same idiom
        # Inductor uses for vanilla slice/transpose: the buffer keeps its
        # canonical PyTorch strides; the load address pattern is encoded in
        # the view's non-canonical stride vector resolved via ``make_loader``.
        # See ``insert_restickify_padding_analysis.md`` §3.
        #
        # The permutation we need is already encoded in ``op.layout``: its
        # ``size`` and ``stride`` describe a view onto the input view's
        # storage with the same axis ordering the clone preserved.  We reuse
        # that pattern by (a) finding the output axis whose extent matches
        # the input's padded host dim (= the within-stick axis on the
        # output's view onto input), (b) bumping that axis's extent from
        # ``host_size[new_stick_dim]`` to ``padded_size[new_stick_dim]``,
        # and (c) rescaling neighboring strides.
        #
        # ``op.data.ranges`` stays at the user's logical output extent
        # (== ``op.layout.size``); widening ranges to the stick-padded extent
        # was tried but inductor's _simplify_loops fuses the widened iter
        # back, so the OpSpec ends up unwidened anyway.
        old_pw = op.data

        out_size = [int(concretize_expr(s)) for s in op.get_layout().size]
        out_stride = [int(concretize_expr(s)) for s in op.get_layout().stride]

        # The output axis corresponding to the input's padded host dim is the
        # one whose extent matches the input's pre-pad extent at new_stick_dim.
        # ``op.layout`` already encodes the permutation that mapped input
        # host axes to output host axes; we just need the index.
        orig_extent = host_size[new_stick_dim]
        padded_extent = padded_size[new_stick_dim]
        out_axis_candidates = [i for i, s in enumerate(out_size) if s == orig_extent]
        if len(out_axis_candidates) != 1:
            logger.warning(
                "insert_restickify_padding: could not locate output axis for "
                "input dim %d (extent=%d, out_size=%s); skipping",
                new_stick_dim,
                orig_extent,
                out_size,
            )
            continue
        out_padded_axis = out_axis_candidates[0]

        # Build the view: bump out_size at the padded axis, then recompute
        # strides in the same axis-order as op.layout (dim_order = axes
        # sorted by descending stride, outermost first).
        view_size = list(out_size)
        view_size[out_padded_axis] = padded_extent
        dim_order = sorted(range(len(out_stride)), key=lambda i: -out_stride[i])
        view_stride = [0] * len(view_size)
        running = 1
        for ax in reversed(dim_order):
            view_stride[ax] = running
            running *= view_size[ax]

        # The view is a transposed reinterpretation of ``padded_buf``.  Use
        # FixedTiledLayout (not plain FixedLayout) for type uniformity with the
        # rest of the post-pre-scheduling graph: every other layout there is
        # tiled with a concrete device_layout, and any layout-aware downstream
        # consumer would silently fall back to the wrapped buffer's STL on a
        # plain FixedLayout view.  The STL is derived from (view_size,
        # view_stride) with dim_order = non-stick host dims in natural order,
        # within-stick host dim last.  The within-stick host dim is the one
        # with view stride 1 (mirrors _build_layout_preserving_padded_stl).
        within_stick_host_dim = next(i for i, s in enumerate(view_stride) if s == 1)
        view_dim_order = [
            i for i in range(len(view_size)) if i != within_stick_host_dim
        ] + [within_stick_host_dim]
        view_stl = SpyreTensorLayout(view_size, view_stride, dtype, view_dim_order)
        view_layout = FixedTiledLayout(
            device,
            dtype,
            size=view_size,
            stride=view_stride,
            device_layout=view_stl,
        )
        view = ReinterpretView(data=padded_buf, layout=view_layout)
        view_loader = view.make_loader()

        new_pw = Pointwise(
            device=old_pw.device,
            dtype=old_pw.dtype,
            inner_fn=lambda index, _loader=view_loader: _loader(index),
            ranges=old_pw.ranges,
        )
        # Announce intent to the codegen-side classifier: after Inductor's
        # _apply_loop_reordering runs, the iter-axis ordering of this op may
        # collapse so input/output device coords share the same free symbol
        # on the within-stick axis, defeating the structural detection in
        # spyre_kernel.py. The tag preserves RESTICKIFY_OP classification so
        # #2112's ceil-div + backGap padded-read path stays gated correctly.
        # Set BEFORE replace_computed_buffer_body so copy_op_metadata carries
        # the attribute onto the replacement ComputedBuffer.
        op._spyre_force_restickify = True
        replace_computed_buffer_body(op, new_pw, operations)
        logger.info(
            "insert_restickify_padding: padded %s dim[%d]: %d -> %d",
            in_dep.name,
            new_stick_dim,
            n,
            n + pad,
        )
