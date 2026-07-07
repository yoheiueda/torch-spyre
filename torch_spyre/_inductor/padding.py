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


def _single_free_sym(expr: Expr):
    """Return ``expr``'s sole free symbol, or None if it has zero or many."""
    syms = expr.free_symbols
    return next(iter(syms)) if len(syms) == 1 else None


def _device_coords(stl: SpyreTensorLayout, dep) -> list[Expr]:
    """Return device-space coordinate expressions for ``dep`` against ``stl``.

    A padding-pass-local peer of ``pass_utils.host_coordinates`` /
    ``device_coordinates``: it shares the concretize-index + compute_coordinates
    core but deliberately omits ``check_stick_expr_supported``.  The pass
    projects an OUTPUT stick coord through an INPUT dep (cross-layout, see
    ``_project_stick_host_dim``), which composes two stride patterns into
    intermediate stick exprs like ``Mod(d2, 32)`` or ``2*(Mod(d1, 32)) + 1``
    that are perfectly valid for the pass's free-*variable* analysis but are not
    codegen-legal stick forms.  Only the free symbols matter here, so the
    codegen validation would wrongly reject a real restickify candidate.

    Returns ``[]`` for a scalar / zero-dim layout (empty ``device_size``).
    """
    index = concretize_index(dep.index, set(dep.ranges.keys()))
    return compute_coordinates(stl.device_size, stl.stride_map, dep.ranges, index)


def _named_write_dep(op):
    """Return ``op``'s sole named write dep.

    Every ComputedBuffer on the restickify path (the candidate itself, or a
    ComputedBuffer producer) has exactly one named write.  Fetch it loudly: an
    empty writes set is an invariant violation, not a shape we silently skip.
    """
    return next(d for d in op.get_read_writes().writes if hasattr(d, "name"))


def _host_dim_carrying_sym(host_coords: list[Expr], sym) -> int | None:
    """Return the outermost host dim whose coordinate carries ``sym``, or None.

    A loop symbol whose range straddles a tile boundary can appear in several
    coordinates; the outermost (lowest-index) one is the governing dim, so we
    return the first match.
    """
    for dim, coord in enumerate(host_coords):
        if sym in coord.free_symbols:
            return dim
    return None


def _host_dim_for_stick_sym(host_coords: list[Expr], sym, sizes: list) -> int | None:
    """Return the input host dim carrying stick symbol ``sym``, or None.

    Shared tail of the two stick-projection helpers: once a within-stick symbol
    has been isolated (from either the source layout's device coord or the op's
    own write dep), the input host dim carrying it is the same lookup.  A
    symbol-free stick (``sym is None``) means a size-1 host dim occupies the
    stick, so fall back to a size-1 host dim (see the branch comment for why any
    one is safe when several are size-1).
    """
    if sym is None:
        # A symbol-free stick means a size-1 host dim moved into stick position.
        # With >=2 size-1 host dims the choice is ambiguous by symbol, but it is
        # a zero-extent relabel: size-1 dims must not contribute to the device
        # layout (tensors_and_layouts.md canonical-form rule), so every size-1
        # dim maps to host_size 1 and the physical dim to pad is re-derived from
        # device-side markers, not from this host index (see
        # _restickify_input_device_dim / _restickify_output_device_dim).  So any
        # size-1 dim yields the same device layout -- pick the first.  A genuine
        # device-level ambiguity (>=2 size-1 device dims) still declines there.
        ones = [i for i, s in enumerate(sizes) if concretize_expr(s) == 1]
        return ones[0] if ones else None
    return _host_dim_carrying_sym(host_coords, sym)


def _project_stick_host_dim(
    input_layout: FixedTiledLayout, stick_source_layout: FixedTiledLayout, dep
) -> int | None:
    """Return the input_layout host dim carrying stick_source_layout's
    within-stick coord under dep, or None if it has no single free variable.

    When the two layouts are the same this is the buffer's own within-stick
    host dim; when they differ, stick_source_layout's stick coordinate is
    projected through dep.  The projection is by free *variable*: the host dim
    whose coordinate carries the stick coord's symbol.  The caller then reads a
    restickify off ``in_stick_dim != new_stick_dim`` — the same test codegen
    uses (``in_coords[-1].free_symbols != out_coords[-1].free_symbols``,
    spyre_kernel.py).  A constant coefficient or offset from a sliced stick
    device-dim size (e.g. ``2*(Mod(var, 32)) + 1``) does not change the free
    variable, so such rescaled coords need no shape special-case.

    When the stick coord is symbol-free -- a size-1 host dim occupies the stick
    (coord collapses to a constant 0) -- there is no symbol to project, so fall
    back to matching the sole size-1 host dim by size (declining if the match is
    ambiguous).  This lets a transpose that moves a size-1 dim into stick
    position be recognised as a restickify.
    """
    host_coords = host_coordinates(input_layout, dep, None)
    device_coords = _device_coords(stick_source_layout.device_layout, dep)
    # No coords means a scalar / zero-dim layout: no stick dim to project.
    if not host_coords or not device_coords:
        return None
    sym = _single_free_sym(device_coords[-1])
    return _host_dim_for_stick_sym(host_coords, sym, list(input_layout.size))


def _output_stick_symbol(op, out_layout):
    """Return the OUTPUT within-stick iteration symbol, or None if symbol-free.

    The symbol comes from the op's own write dep against the output stl -- a
    clean single symbol -- unlike a cross-layout projection through the input
    dep, which composes two stride patterns into a multi-symbol coord whenever
    both stick dims are sub-64 and alias the same physical region.
    """
    write_dep = _named_write_dep(op)
    out_dev_coords = _device_coords(out_layout.device_layout, write_dep)
    if not out_dev_coords:
        return None
    return _single_free_sym(out_dev_coords[-1])


def _stick_free_symbols(layout: FixedTiledLayout, dep) -> frozenset:
    """Return the free symbols of ``layout``'s within-stick device coord under
    ``dep``, or an empty set if the layout has no device coords.
    """
    device_coords = _device_coords(layout.device_layout, dep)
    if not device_coords:
        return frozenset()
    return frozenset(device_coords[-1].free_symbols)


def _codegen_will_restickify(op, out_layout, in_layout, in_dep) -> bool:
    """Return whether codegen will emit a RESTICKIFY (vs IDENTITY) for ``op``.

    Mirrors the store-side test in spyre_kernel.py exactly: codegen restickifies
    iff the input and output within-stick coords carry different free symbols.
    The pass MUST agree with this predicate -- a restickify codegen emits on an
    unpadded, unaligned buffer over-reads uninitialized stick lanes.  So the pass
    only returns None ("carry on unpadded") when this is False; when it is True
    the op is either padded or refused loudly, never silently skipped.
    """
    in_syms = _stick_free_symbols(in_layout, in_dep)
    out_write_dep = _named_write_dep(op)
    out_syms = _stick_free_symbols(out_layout, out_write_dep)
    return in_syms != out_syms


def _output_stick_input_host_dim(op, out_layout, in_layout, in_dep) -> int | None:
    """Return the input host dim carrying the OUTPUT stick's iteration symbol.

    Mirrors codegen (spyre_kernel.py): the output within-stick symbol comes from
    the output layout's own write dep, then is located among the input's host
    coords via the read dep.  A symbol-free output stick means a size-1 host dim
    moved into stick position; fall back to matching the sole size-1 dim.
    """
    out_stick_sym = _output_stick_symbol(op, out_layout)
    in_host_coords = host_coordinates(in_layout, in_dep, None)
    if not in_host_coords:
        return None
    return _host_dim_for_stick_sym(in_host_coords, out_stick_sym, list(in_layout.size))


def _identify_restickify_candidate(op: Operation, graph: GraphLowering):
    """Identify whether ``op`` is a restickify the padding pass may act on.

    A candidate is a single-input pointwise copy between two FixedTiledLayouts
    that lands a *different* host dim within the stick.

    Returns None if ``op`` is not a candidate, else the tuple:

    - ``in_dep`` / ``in_buf`` / ``in_layout``: the single input's dep, buffer,
      and (FixedTiled) layout.
    - ``host_size``: the input's concretized host size.
    - ``new_stick_dim``: input host dim that becomes the output's stick dim.
    - ``in_stick_dim``: input host dim that becomes the output's "old-stick"
      non-stick device dim.
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
    new_stick_dim = _output_stick_input_host_dim(op, out_layout, in_layout, in_dep)
    if in_stick_dim is None:
        return None
    if new_stick_dim is None:
        # No input host dim was located for the output stick.  This is safe to
        # skip only if codegen will NOT restickify (identical in/out stick
        # symbols -> IDENTITY, no over-read).  If codegen WILL restickify, the
        # buffer reaches codegen unpadded and over-reads uninitialized stick
        # lanes -- so fail loudly rather than miscompile.  Reaching here with a
        # real restickify means the output stick carries an iteration symbol that
        # no input host dim carries (e.g. a fused multi-symbol coord); a size-1
        # output stick always resolves to a host dim (_host_dim_for_stick_sym
        # picks the first size-1 dim), so it does not land here.
        if _codegen_will_restickify(op, out_layout, in_layout, in_dep):
            raise Unsupported(
                "restickify padding: cannot locate input host dim for output stick"
            )
        return None
    if new_stick_dim == in_stick_dim:
        return None

    host_size = [concretize_expr(s) for s in in_layout.size]

    # No over-read when the output stick aliases the input's OWN stick.  Projecting
    # the output stick coord through the input read dep (the pre-3ddc683 method)
    # lands back on ``in_stick_dim``: the restickified read stays inside the input's
    # already-initialized stick, so no padding is needed -- and padding it would
    # relabel a tracked named dim (see test_permute_matmul_distinct_lqlk, a permuted
    # matmul input whose full D=64 stick re-tiles to a sub-64 Lk output stick).
    # This is distinct from the 3ddc683 sub-stick->sub-stick hazard, where the
    # projection composes two sub-64 stride patterns into a multi-symbol coord and
    # returns None: that case must stay a candidate, so gate on "not None".  A
    # genuine unaligned transpose projects onto a *different* host dim
    # (projection != in_stick_dim), so it is kept and padded.
    projected = _project_stick_host_dim(in_layout, out_layout, in_dep)
    if projected is not None and projected == in_stick_dim:
        return None

    # The read-side padding paths (producer-grow, identity-clone copy) read the
    # input verbatim, so they can only own a restickify whose read is a pure
    # permutation.  A host dim iterated with a range wider than its size is a
    # broadcast (e.g. the qkv rope size-2 dim read 128x): there is no unique
    # input dim to grow and no verbatim read to redirect.  Decline quietly and
    # leave it to codegen, as before the output-write-dep derivation promoted it
    # to a candidate.  A genuine transpose reads every dim over its full size, so
    # this never drops a real hazard; a narrowing slice (iter range < dim size)
    # still falls through to the copy path's loud guard.
    in_host_coords = host_coordinates(in_layout, in_dep, None)
    for i, coord in enumerate(in_host_coords):
        sym = _single_free_sym(coord)
        if sym is None:  # degenerate size-1 host dim, nothing to iterate
            continue
        if concretize_expr(in_dep.ranges[sym]) > concretize_expr(in_layout.size[i]):
            return None

    return in_dep, in_buf, in_layout, host_size, new_stick_dim, in_stick_dim


def _device_dim_carrying_sym(stl: SpyreTensorLayout, write_dep, sym) -> int | None:
    """Return the outermost non-within-stick ``device_size`` dim whose device
    coordinate carries ``sym`` (the free symbol of some host dim), or None.

    The dim is located by symbol match, not dim-size equality: two host dims can
    share a dim size, and only the symbol is unambiguous.  A loop symbol whose
    range straddles a tile boundary can appear in several coordinates; the
    outermost (lowest-index) one is the governing dim, so we return the first
    match.  The within-stick dim (the last device coordinate) is excluded.
    """
    device_coords = _device_coords(stl, write_dep)
    for dim in range(len(device_coords) - 1):
        if sym in device_coords[dim].free_symbols:
            return dim
    return None


def _restickify_output_device_dim(
    op: ComputedBuffer, in_dep, in_layout, in_stick_dim: int
) -> int | None:
    """Return the device-size dim index of the non-stick device dim that carries
    the input's old stick dim, or None if it is stick-aligned and needs no
    padding.

    That dim is the host dim carrying the iter symbol of the input's old stick
    dim (``in_stick_dim``).  After the restickify it is a non-stick device dim
    whose true dim size is the (small) old-stick host size.  Padding is required
    whenever that device dim's size is not a stick multiple, regardless of how
    many stick blocks the new stick dim spans or where the block dim sits
    relative to this dim: bumping the dim to a stick boundary widens the physical
    allocation so every batch plane and stick block lands at the correct offset.
    """
    # The old stick dim's host coord carries more than one iter symbol (a fused
    # host dim): no single symbol to track through to a device dim, so decline.
    in_host_coords = host_coordinates(in_layout, in_dep, None)
    old_sym = _single_free_sym(in_host_coords[in_stick_dim])
    if old_sym is None:
        return None
    # Old stick dim collapsed to a size-1 output host dim (const-0 coord, no
    # symbol): nothing survives to misalign, so nothing to pad.
    out_layout = op.get_layout()
    write_dep = _named_write_dep(op)
    out_host_coords = host_coordinates(out_layout, write_dep, None)
    if _host_dim_carrying_sym(out_host_coords, old_sym) is None:
        return None
    stl = out_layout.device_layout
    device_dim = _device_dim_carrying_sym(stl, write_dep, old_sym)
    if device_dim is None:
        return None
    # Already a stick multiple: stick blocks land aligned, no padding needed.
    stick_size = get_elem_in_stick(out_layout.dtype)
    if stl.device_size[device_dim] % stick_size == 0:
        return None
    return device_dim


def _pad_layout_device_dim(
    layout: FixedTiledLayout,
    device_dim: int,
    new_dim_size,
    grow_host_dim: int | None = None,
) -> FixedTiledLayout:
    """Return a copy of ``layout`` with one ``device_size`` dim grown to
    ``new_dim_size``, and ``host_size[grow_host_dim]`` grown to match when
    ``grow_host_dim`` is set (leaving the host size logical when it is None).

    ``stride_map`` and ``host_stride`` are both left untouched.  A ``stride_map``
    entry is the *per-step* host stride along a device dim, and growing a dim's
    extent does not change how far one step moves — the same way a plain slice
    changes ``device_size`` (its extent) while every ``stride_map`` entry keeps
    pointing at the same host element spacing.  The identity
    ``host_offset = dot(device_coordinates, stride_map)`` therefore still maps
    device coordinates onto the same storage; only the allocation is wider.
    """
    stl = layout.device_layout
    new_device_size = list(stl.device_size)
    new_device_size[device_dim] = new_dim_size
    padded_stl = SpyreTensorLayout(
        new_device_size, list(stl.stride_map), stl.device_dtype, stl.element_arrangement
    )
    host_size = [concretize_expr(s) for s in layout.size]
    host_stride = [concretize_expr(s) for s in layout.stride]
    if grow_host_dim is not None:
        host_size[grow_host_dim] = new_dim_size
    return FixedTiledLayout(
        layout.device, layout.dtype, host_size, host_stride, padded_stl
    )


def _restickify_output_size1_device_dim(
    op: ComputedBuffer, in_dep, in_layout, in_stick_dim: int
) -> int | None:
    """Return the collapsed old-stick output device dim when the input's old
    stick host dim is size 1, or None when it does not apply / is ambiguous.

    Companion to ``_restickify_output_device_dim`` for the case that function
    declines with ``old_sym is None``: a size-1 input old-stick host dim carries
    no symbol to project, so the output dim it collapses to is a size-1 singleton
    (``device_size == 1``) marked ``stride_map == -1``.  Mirrors the input-side
    ``_restickify_input_device_dim`` tiebreak; declines if still ambiguous.

    Unlike the aligned path this dim cannot be grown in place here: growing a
    size-1 device dim before ``align_tensors`` yields a fractional coordinate
    (``7*c0/64``) that ``normalize_coordinates`` rejects.  The scheduler grows it
    after align instead (see ``_grow_size1_stick_allocations``); this function
    only locates the dim to tag.

    When the input has a SECOND size-1 host dim (a leading/middle extra size-1
    dim alongside the real innermost stick), the output has two-or-more device
    dims that are size-1 with ``stride_map == -1``, and the sole ``-1`` marker no
    longer isolates the collapsed old stick.  Only one of them is the demoted old
    stick; the other(s) are incidental input size-1 dims that never carried stick
    data.  They are told apart by geometry: the batch / preserved axes (the
    "plane" dims the transpose leaves in place) carry an iteration symbol
    (range > 1), while the old stick and the incidental size-1 dims are
    symbol-free.  In the device row-major ordering the demoted old stick lands at
    the slot FARTHEST from the plane axes -- adjacent to the stick block when the
    batch nest is outer, or at the outermost slot when the batch nest is inner --
    whereas an incidental size-1 dim sits among the batch axes.  Picking the
    ``-1`` candidate that maximises its minimum distance to any plane dim
    therefore recovers the same dim the N>=2 sibling (extra size-1 dim grown to
    size 2) marks as its sole ``-1``.  This is also the dim
    ``_grow_size1_stick_allocations`` grows with per-step stride
    ``prod(host_size)`` -- the outermost-varying non-stick axis.
    """
    in_host_coords = host_coordinates(in_layout, in_dep, None)
    if _single_free_sym(in_host_coords[in_stick_dim]) is not None:
        return None  # not a size-1 old-stick dim; aligned path owns it
    out_layout = op.get_layout()
    stl = out_layout.device_layout
    device_size = [concretize_expr(s) for s in stl.device_size]
    stride_map = list(stl.stride_map)
    size1 = [d for d in range(len(device_size) - 1) if device_size[d] == 1]
    grow = [d for d in size1 if stride_map[d] == -1]
    if len(grow) == 1:
        return grow[0]
    if not grow:
        return size1[0] if len(size1) == 1 else None
    # Two-or-more collapsed size-1 device dims: the demoted old stick is the one
    # farthest from the plane (batch/preserved) dims.  Plane dims are the
    # non-stick device dims whose coordinate carries a free iteration symbol.
    write_dep = _named_write_dep(op)
    dev_coords = _device_coords(stl, write_dep)
    plane = [
        d
        for d in range(len(device_size) - 1)
        if _single_free_sym(dev_coords[d]) is not None
    ]
    if not plane:
        # All-ones batch (e.g. [1,1,64,1]): no plane to measure against, and every
        # candidate is a zero-extent relabel, so pick-first stays byte-correct.
        return grow[0]
    return max(grow, key=lambda g: (min(abs(g - p) for p in plane), -g))


def _pad_restickify_output(
    op: ComputedBuffer, in_dep, in_layout, in_stick_dim: int
) -> None:
    """Pad the output's unaligned non-stick device dim (the one carrying the
    input's old stick dim) to a stick boundary so the second+ stick block lands
    at the correct offset.

    Only the device layout grows (see _pad_layout_device_dim); the tail rows are
    covered by ``_create_sdsc_tensors``'s backGap path and never read back.

    When the input old-stick host dim is size 1 the output dim collapses to a
    size-1 singleton whose in-place grow would break ``align_tensors``; tag it
    with ``_size1_stick_alloc_dim`` for the scheduler to grow after align.
    """
    device_dim = _restickify_output_device_dim(op, in_dep, in_layout, in_stick_dim)
    if device_dim is None:
        size1_dim = _restickify_output_size1_device_dim(
            op, in_dep, in_layout, in_stick_dim
        )
        if size1_dim is not None:
            op._size1_stick_alloc_dim = size1_dim
            logger.debug(
                "insert_restickify_padding: tagged size-1 output %s device dim %d "
                "for scheduler-window alloc grow",
                op.get_name(),
                size1_dim,
            )
        return

    out_layout = op.get_layout()
    old_dim_size = out_layout.device_layout.device_size[device_dim]
    new_dim_size = old_dim_size + compute_padding(old_dim_size, out_layout.dtype)

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
    dim ``new_stick_dim`` (the dim the restickify turns into its new stick dim),
    or None if the producer geometry does not expose it as a bumpable non-stick
    device dim.

    The restickify over-reads this dim to the stick boundary; bumping the
    producer's device_size on the matching dim makes the producer allocate
    (and its backGap path leave defined) the widened tail, so the over-read
    lands inside the producer's own buffer instead of uninitialised HBM.

    When ``new_stick_dim`` is a size-1 host dim (symbol-free coord), there is no
    symbol to project onto a device dim, so fall back to the sole size-1
    (singleton) producer device dim.  Several device dims can be size-1 (the
    host singleton plus a single-block stick tile-count); disambiguate by the
    ``stride_map == -1`` marker that ``coarse_tile._resize_stl_device_dims``
    uses for the singleton being grown, and decline if still ambiguous.
    """
    layout = producer.get_layout()
    write_dep = _named_write_dep(producer)
    host_coords = host_coordinates(layout, write_dep, None)
    stl = layout.device_layout
    sym = _single_free_sym(host_coords[new_stick_dim])
    if sym is not None:
        return _device_dim_carrying_sym(stl, write_dep, sym)
    # Size-1 host dim: match the singleton producer device dim by size, using
    # stride_map == -1 to break ties among multiple size-1 device dims.
    device_size = list(stl.device_size)
    stride_map = list(stl.stride_map)
    size1_dims = [
        d for d in range(len(device_size) - 1) if concretize_expr(device_size[d]) == 1
    ]
    grow = [d for d in size1_dims if stride_map[d] == -1]
    if len(grow) == 1:
        return grow[0]
    return size1_dims[0] if len(size1_dims) == 1 else None


def _pad_restickify_input_via_producer(
    in_buf: ComputedBuffer, new_stick_dim: int
) -> bool:
    """Grow a producer's output to the stick-aligned dim size so the restickify
    reads its widened tail directly, avoiding the ``lower_pad_sequence`` copy
    (separate buffer + fill + copy + HBM round-trip).

    Grows the producer's device_size AND host_size on the dim carrying
    ``new_stick_dim`` to the stick boundary (host_stride and stride_map are
    unchanged: growing the outermost extent does not alter the per-step strides
    -- see _pad_layout_device_dim).  The restickify then over-reads the widened
    device dim, which now lands inside the producer's own (wider) allocation
    instead of uninitialised HBM.

    The grow is layout-only (device_size + host_size on one dim), so it does not
    branch on the producer's op kind.  The producer may be a Pointwise or a
    Reduction (e.g. a sliced matmul output); the two differ in what the grown
    tail contains, but both are safe:

    - Pointwise: iteration_space follows the output (write.ranges), so growing
      host_size grows the iteration space and the producer writes real values
      into the widened tail.
    - Reduction (matmul): iteration_space follows the input (read.ranges, the
      K-space), so growing the output dim does NOT grow the iteration space --
      the producer's own write gets dev_dim_size > it_dim_size and its
      write-side backGap leaves the tail unwritten (garbage).

    Either way the tail never reaches a read position: every consumer iterates
    its own it_dim_size, capped at the producer's logical extent, so a plain,
    reducing, or restickifying co-reader stops before the bumped tail (the same
    per-consumer bound that lets the restickify output discard its own padded
    rows).  A grown buffer also cannot be LX-pinned -- device_size > it_dim_size
    trips the allocator's back-gap gate (_would_produce_lx_back_gap), which the
    backend supports on HBM but not LX -- so the fusion needs no consumer guard.

    The one invariant the grow relies on is that the producer carries a
    ``FixedTiledLayout``: the caller (_identify_restickify_candidate) requires it
    here, and codegen's store() requires it again downstream.  This is the real
    precondition, not the op kind -- an op-kind guard would be both narrower than
    needed (Scatter is a Pointwise subclass, the *Reduction variants subclass
    Reduction) and a false precondition (Scan/Sort would trip it though the
    layout-only grow stays correct).

    Two producer kinds need no guard because the caller never reaches them: the
    caller already requires the input's layout to be a ``FixedTiledLayout``
    (_identify_restickify_candidate), so a mutation-layout producer never
    arrives here (and would be discarded anyway, since propagate_mutation_layouts
    reassigns such layouts after this pass); and a producer that is also a graph
    output is never restickified in place, since Inductor realizes such a graph
    so the restickify reads the underlying source, not the output buffer.
    """
    name = in_buf.get_name()

    device_dim = _restickify_input_device_dim(in_buf, new_stick_dim)
    if device_dim is None:
        return False

    layout = in_buf.get_layout()
    old_dim_size = layout.device_layout.device_size[device_dim]
    n = concretize_expr(layout.size[new_stick_dim])
    new_dim_size = n + compute_padding(n, layout.dtype)

    # Bump the device_size dim; unlike the output case the producer's host dim
    # also grows to new_dim_size (grow_host_dim) so it actually computes the
    # widened tail (see _pad_layout_device_dim).
    in_buf.layout = _pad_layout_device_dim(
        layout, device_dim, new_dim_size, grow_host_dim=new_stick_dim
    )

    logger.debug(
        "insert_restickify_padding: fused pad into producer %s device dim %d "
        "%d -> %d (host dim %d: %d -> %d)",
        name,
        device_dim,
        old_dim_size,
        new_dim_size,
        new_stick_dim,
        n,
        new_dim_size,
    )
    return True


def lower_identity_clone(
    arg_fx_node: torch.fx.Node,
    host_size: list[int],
    device: torch.device,
    dtype: torch.dtype,
    orig_stl: SpyreTensorLayout,
    insert_before: torch.fx.Node,
) -> tuple[ComputedBuffer, list[Operation]]:
    """Lower an identity ``aten.clone`` of ``arg_fx_node`` (peer to
    ``lower_pad_sequence``).

    Unlike ``lower_pad_sequence`` this allocates a buffer at the ORIGINAL
    unpadded ``host_size`` and emits a single copy op -- no fill constant, no
    fill-region mutation.  The clone's host geometry is identical to the input,
    so its ``SpyreTensorLayout`` mirrors ``orig_stl`` verbatim; the caller bumps
    ``device_size`` on the stick-carrying dim afterwards (keeping this helper
    generic).

    ``insert_restickify_padding`` runs after ``propagate_spyre_tensor_layouts``,
    so a ``run_node``-lowered op keeps a ``FlexibleLayout`` unless a
    ``FixedTiledLayout`` is assigned immediately -- done here.

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
    # a named ComputedBuffer (mirrors lower_restickify at lowering.py) rather than
    # inlining into the consumer.
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
    clone_buf.layout = FixedTiledLayout(
        host_layout.device,
        host_layout.dtype,
        host_layout.size,
        host_layout.stride,
        clone_stl,
    )

    # LX planning reads origin_node directly on the ComputedBuffer.
    object.__setattr__(clone_buf, "origin_node", clone_fx)
    assert clone_buf.origins, "lower_identity_clone: clone buffer has no origins"

    return clone_buf, new_ops


def _assert_input_not_sliced(op: ComputedBuffer, in_dep, in_layout) -> None:
    """Raise ``Unsupported`` when any input host dim is sliced (iter range < dim
    size).

    The unified copy path redirects the restickify to read an identity clone of
    the input verbatim -- it does not re-index -- so a sliced input would
    silently over-read the untouched dim.  A genuine restickify whose moved
    stick dim or any other dim is sliced cannot be padded this way, so fail
    loudly instead.  A plain pointwise stick-dim slice never reaches here (the
    candidate matcher declines it: same in-/new-stick dim).
    """
    in_host_coords = host_coordinates(in_layout, in_dep, None)
    for i, coord in enumerate(in_host_coords):
        sym = _single_free_sym(coord)
        if sym is None:  # degenerate size-1 host dim, nothing to slice
            continue
        iter_range = concretize_expr(in_dep.ranges[sym])
        dim_size = concretize_expr(in_layout.size[i])
        if iter_range != dim_size:
            raise Unsupported(
                f"insert_restickify_padding: sliced input on host dim "
                f"{i} of {op.get_name()} (iter range {iter_range} != "
                f"dim size {dim_size}) is not supported"
            )


def _pad_restickify_input_via_copy(
    op: ComputedBuffer,
    operations: list[Operation],
    in_dep,
    in_buf,
    in_layout,
    new_stick_dim: int,
) -> None:
    """Fallback read-side fix: insert an identity clone of the input ahead of the
    restickify and redirect the restickify to read it.

    Used when the input cannot be grown in place by
    ``_pad_restickify_input_via_producer`` (a graph input, whose allocation is
    not ours to widen).  The clone mirrors the producer path: it has the input's
    exact host geometry with ``device_size`` bumped to the stick boundary on the
    new-stick dim, so the restickify's over-read lands inside the clone's own
    allocation.  The bumped tail is written by neither a fill nor the clone's
    store (the identity copy iterates only the real rows), and is only ever read
    into the output's backGap-discarded stick band -- so its contents are
    don't-care, exactly as in the producer path.

    Only one copy op is emitted (no zero-fill), and the redirect reuses the
    canonical ``NameSwapHandler`` rather than rewriting the restickify body.
    """
    device = in_buf.get_device()
    if device is None:
        return

    # Redirect reads the input verbatim; a sliced input would over-read.
    _assert_input_not_sliced(op, in_dep, in_layout)

    dtype = in_layout.dtype
    host_size = [concretize_expr(s) for s in in_layout.size]

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

    # Bump the clone's device_size on the dim carrying the new stick dim to the
    # stick boundary (device_size only; host size/stride/stride_map unchanged).
    # The clone is a ComputedBuffer we own, so its stick-carrying device dim is
    # located the same way as a producer's.
    device_dim = _restickify_input_device_dim(clone_buf, new_stick_dim)
    assert device_dim is not None, (
        f"_pad_restickify_input_via_copy: no device dim carrying new stick dim "
        f"{new_stick_dim} for clone {clone_buf.get_name()}"
    )
    n = host_size[new_stick_dim]
    new_dim_size = n + compute_padding(n, dtype)
    clone_buf.layout = _pad_layout_device_dim(
        clone_buf.get_layout(), device_dim, new_dim_size, grow_host_dim=None
    )

    # Move the clone op to just before the restickify (run_node appends).
    _move_ops_before(operations, new_ops, op)

    # Redirect the restickify to read the clone (wrap-not-reconstruct).
    redirect_computed_buffer_reads(op, {in_dep.name: clone_buf.get_name()}, operations)


def insert_restickify_padding(graph: GraphLowering) -> None:
    """Pad a restickify's buffers so codegen's stick-boundary widening never
    touches uninitialized HBM.

    A restickify re-tiles a tensor so a different host dim lands within the
    stick.  Codegen widens both its read and its write to stick boundaries,
    which exposes two independent hazards, each with its own fix:

    - Write side (``_pad_restickify_output``): when the new stick dim spans
      more than one stick block, the output's old-stick host dim becomes a
      non-stick device dim; if its dim size is not a stick multiple the second+
      block lands at the wrong physical offset.  Always attempted.
    - Read side: when the new stick dim's size is not a stick multiple the read
      runs past the true dim size.  ``_pad_restickify_input_via_producer``
      grows the producer in place when eligible; otherwise
      ``_pad_restickify_input_via_copy`` inserts a device-size-bumped identity
      clone and redirects the restickify to read it.

    The two are orthogonal — e.g. a 128x67 transpose has an aligned input
    stick dim (128) but an unaligned non-stick dim (67), so only the write-side
    fix fires.
    """
    operations = graph.operations
    for op in list(operations):
        match = _identify_restickify_candidate(op, graph)
        if match is None:
            continue
        in_dep, in_buf, in_layout, host_size, new_stick_dim, in_stick_dim = match
        # ComputedBuffer guaranteed by _identify_restickify_candidate
        assert isinstance(op, ComputedBuffer)

        _pad_restickify_output(op, in_dep, in_layout, in_stick_dim)

        if compute_padding(host_size[new_stick_dim], in_layout.dtype) == 0:
            continue

        # Grow the producer in place when the input is one we own (a
        # ComputedBuffer); a graph input's allocation is not ours to widen, so it
        # takes the identity-clone copy path.
        if isinstance(in_buf, ComputedBuffer) and _pad_restickify_input_via_producer(
            in_buf, new_stick_dim
        ):
            continue
        _pad_restickify_input_via_copy(
            op, operations, in_dep, in_buf, in_layout, new_stick_dim
        )
