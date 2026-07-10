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


def _size1_alloc_dim(stl: SpyreTensorLayout, tiebreak=None) -> int | None:
    """Locate the size-1 (singleton) non-stick device dim that a restickify
    grows to a stick, or None if the choice is ambiguous.

    A size-1 host dim that moves into (input side) or out of (output side) stick
    position carries no iteration symbol, so it cannot be tracked to a device dim
    by symbol.  It is instead the singleton (``device_size == 1``) device dim
    marked with ``stride_map == -1`` (the extent-1 / never-stepped marker
    handled by ``coarse_tile._resize_device_layout``).
    Several device dims can be size-1 (the host singleton plus a single-block
    stick tile-count), so:

    - a sole ``-1`` marker isolates the dim -> return it;
    - no ``-1`` marker -> fall back to a sole size-1 dim, else decline;
    - two-or-more ``-1`` markers -> hand the candidates to ``tiebreak`` (the
      output side breaks the tie by geometry; the input side passes None and
      declines).
    """
    device_size = [concretize_expr(s) for s in stl.device_size]
    stride_map = list(stl.stride_map)
    size1 = [d for d in range(len(device_size) - 1) if device_size[d] == 1]
    grow = [d for d in size1 if stride_map[d] == -1]
    if len(grow) == 1:
        return grow[0]
    if not grow:
        return size1[0] if len(size1) == 1 else None
    return tiebreak(grow) if tiebreak is not None else None


def _dep_range(dep, sym) -> int | None:
    """Return the concretized iteration range of loop symbol ``sym`` on ``dep``,
    or None if ``sym`` is None or absent from the dep's ranges.

    The read's iteration range over a dim is that dim's true logical fill -- 64
    for a full stick dim, 2/48 for a sub-stick one -- which is the quantity the
    restickify padding decisions turn on ("is this dim already a whole stick?").
    Reading it off the dep keys off the same loop symbol codegen uses instead of
    re-projecting a stick back to a host dim and re-indexing ``host_size``; the
    device extent cannot substitute (within-stick device dims are stick-rounded
    to 64 and cannot distinguish a full stick from a padded sub-stick one).
    """
    if sym is None or sym not in dep.ranges:
        return None
    return concretize_expr(dep.ranges[sym])


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
        # _restickify_input_device_dim / _pad_restickify_output).  So any
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
    uses (the ``is_restickify`` predicate in pass_utils, which compares the
    within-stick coords' free symbols).  A constant coefficient or offset from a
    sliced stick
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


def _codegen_will_restickify(op, out_layout, in_layout, in_dep) -> bool:
    """Return whether codegen will emit a RESTICKIFY (vs IDENTITY) for ``op``.

    Delegates to the shared ``is_restickify`` predicate (pass_utils) that codegen
    itself calls, so the pass and the store side cannot drift apart.  Used only by
    the can't-happen backstop in ``_identify_restickify_candidate``: when no input
    host dim carries the output stick, this decides skip (codegen won't restickify)
    vs. refuse loudly (it would, so the unpadded buffer would over-read
    uninitialized stick lanes).

    The pass supplies the two operands' device coords via ``_device_coords`` (the
    padding-local peer that omits ``check_stick_expr_supported`` -- only the free
    symbols matter here); codegen supplies the concrete ``device_coordinates`` of
    its tensor args.  Both feed the same predicate.
    """
    in_coords = _device_coords(in_layout.device_layout, in_dep)
    out_coords = _device_coords(out_layout.device_layout, _named_write_dep(op))
    if not in_coords or not out_coords:
        return False
    return is_restickify(in_coords, out_coords)


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
    that lands a *different* host dim within the stick.  After confirming the
    shape (pointwise, one input, both FixedTiled), it derives the two host dims
    the padding logic needs and applies two gates that reject non-candidates:
    the restickify test does not hold (same stick dim), or the read is a
    broadcast.  A third branch is a can't-happen backstop: a recognized
    restickify whose output stick maps to no input host dim is refused loudly
    rather than skipped (see below).

    Returns None if ``op`` is not a candidate, else the tuple:

    - ``in_dep`` / ``in_buf`` / ``in_layout``: the single input's dep, buffer,
      and (FixedTiled) layout.
    - ``new_stick_dim``: input host dim that becomes the output's stick dim.
    - ``in_stick_dim``: input host dim that becomes the output's "old-stick"
      non-stick device dim.

    Both are indices into the INPUT host dims (named for what they become on
    the output), so they are directly comparable -- ``new_stick_dim ==
    in_stick_dim`` is the not-a-restickify test.  Neither is an output dim
    index: a restickify re-tiles rather than preserving ranks, so the output
    dim carrying the new stick generally sits at a different index.
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

    # The restickify test, expressed as host dims: the input's own within-stick
    # dim vs. the input dim that carries the output's stick symbol.  A restickify
    # lands a *different* host dim in the stick, i.e. new_stick_dim != in_stick_dim
    # (the free-symbol inequality is_restickify checks, projected onto host dims).
    # These two host dims are the restickify test re-expressed in host-dim space
    # (is_restickify compares the within-stick device symbols; the projection maps
    # each to the input host dim carrying it).  Every skip below therefore means
    # "not a restickify", which must agree with the device-space is_restickify
    # codegen keys off -- else the pass would skip an op codegen still restickifies
    # on an unpadded buffer (over-reading uninitialized stick lanes).  Agreement
    # rests on the single-symbol stick invariant finalize_layouts establishes:
    #
    #   - in_stick_dim is None: the input stick coord has no single symbol.  A
    #     symbol-free stick means a size-1 host dim in stick position (canonical
    #     form), so the size-1 fallback resolves it; None with a real restickify
    #     would need a symbol-free stick AND no size-1 host dim -- inconsistent.
    #   - new_stick_dim == in_stick_dim: a restickify carries two DIFFERENT stick
    #     symbols (is_restickify's inequality), so equal host dims would need one
    #     input host coord to carry both -- the invariant gives one symbol per
    #     stick dim, so it does not arise.
    #
    # Only new_stick_dim is None keeps a live is_restickify assertion, because it
    # is the one path where the projection failing could plausibly hide a real
    # restickify if the invariant ever broke (verified unreachable across the
    # restickify + propagate suite for all three paths).
    in_stick_dim = _project_stick_host_dim(in_layout, in_layout, in_dep)
    new_stick_dim = _output_stick_input_host_dim(op, out_layout, in_layout, in_dep)
    if in_stick_dim is None:
        return None
    if new_stick_dim is None:
        # If the invariant ever breaks and codegen would still restickify, the
        # unpadded buffer over-reads uninitialized lanes -- refuse loudly, not skip.
        if _codegen_will_restickify(op, out_layout, in_layout, in_dep):
            raise Unsupported(
                "restickify padding: output stick maps to no input host dim "
                f"for {op.get_name()} (unexpected: layout invariant broken)"
            )
        return None
    if new_stick_dim == in_stick_dim:
        return None

    return in_dep, in_buf, in_layout, new_stick_dim, in_stick_dim


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


def _pad_layout_device_dim(
    layout: FixedTiledLayout,
    device_dim: int,
    new_dim_size,
    grow_host_dim: int | None = None,
) -> FixedTiledLayout:
    """Return a copy of ``layout`` with one ``device_size`` dim grown to
    ``new_dim_size``, and ``host_size[grow_host_dim]`` grown to match when
    ``grow_host_dim`` is set (leaving the host size logical when it is None).

    ``stride_map`` and ``host_stride`` are both left untouched: a ``stride_map``
    entry is the *per-step* host stride along a device dim, and growing a dim's
    extent does not change how far one step moves.  The identity
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


def _pad_restickify_output(
    op: ComputedBuffer, in_dep, in_layout, in_stick_dim: int
) -> None:
    """Pad the output dim carrying the input's old stick dim to a stick boundary,
    so the second+ stick block and every batch plane land at the correct offset.

    Only the device layout grows (see ``_pad_layout_device_dim``); the tail rows
    are covered by ``_create_sdsc_tensors``'s backGap path and never read back.
    The dim carries the free symbol of ``in_stick_dim``'s host coord; padding is
    needed whenever its device size is not a stick multiple.

    Two shapes of that dim, split on whether the old stick host dim carries a
    symbol at all:

    - **Size 1** (no symbol): the old stick collapses to a size-1 singleton
      device dim, located by ``_size1_alloc_dim`` from the ``stride_map == -1``
      marker (``_old_stick_size1_dim`` breaks a multi-marker tie by the new
      stick's rank in the input device coords, mirroring the SDSC codegen's
      ``_restore_elided_restickify_stick`` insert position).
      It cannot grow in place -- a size-1 grow before ``align_tensors`` yields a
      fractional coordinate ``normalize_coordinates`` rejects -- so tag it with
      ``_size1_stick_alloc_dim`` for the scheduler to grow after align (see
      ``_grow_size1_stick_allocations``).
    - **Has a symbol**: track it to the non-stick device dim and, if unaligned,
      grow the device layout in place.  A fused old-stick host coord (>1 symbol),
      one that collapses to a const-0 output coord, or an already-aligned dim
      needs no padding.
    """
    in_host_coords = host_coordinates(in_layout, in_dep, None)
    old_sym = _single_free_sym(in_host_coords[in_stick_dim])
    out_layout = op.get_layout()
    write_dep = _named_write_dep(op)
    stl = out_layout.device_layout

    if old_sym is None:

        def _old_stick_size1_dim(grow: list[int]) -> int:
            # Two-or-more collapsed size-1 device dims: pick the one that is the
            # demoted old stick to grow to a full stick.
            #
            # The choice mirrors the SDSC codegen exactly.  When codegen restores
            # the elided old stick (``_restore_elided_restickify_stick`` in
            # superdsc.py) it inserts the restored symbol into the OUTPUT device
            # coords at ``new_stick_pos`` -- the rank the NEW stick occupies among
            # the INPUT device coords.  (A transpose swaps the two sticks' slots
            # while every surviving batch/spatial dim keeps its place, so the old
            # stick lands where the new stick used to sit.)  The device dim to grow
            # here is that same slot, so allocation and descriptor agree: the
            # grown 64-wide dim is exactly the one the descriptor writes 64 planes
            # into.
            #
            # This covers both cases with one rule:
            #   - MULTI-BLOCK new stick (host > 64): the output splits it into a
            #     tile-count dim at ``new_stick_pos`` and a within-stick Mod term,
            #     pushing the restored old stick one slot outer -- which is where
            #     ``new_stick_pos`` already points (the input's un-split new-stick
            #     rank), i.e. immediately before the tile-count dim.
            #   - SINGLE-BLOCK new stick: no real tile-count dim, so the old stick
            #     sits directly at ``new_stick_pos``.
            dev_coords = _device_coords(stl, write_dep)
            new_sym = _single_free_sym(dev_coords[-1])
            if new_sym is not None:
                in_dev_coords = _device_coords(in_layout.device_layout, in_dep)
                # Shared rule with codegen's _restore_elided_restickify_stick: the
                # new stick's rank among the INPUT device coords is where the
                # demoted old stick lands.  See pass_utils.restickify_new_stick_pos.
                new_stick_pos = restickify_new_stick_pos(in_dev_coords, {new_sym})
                if new_stick_pos is not None and new_stick_pos in grow:
                    return new_stick_pos
            # Defensive fallback for a new stick with no single symbol (e.g. an
            # all-ones batch whose new stick coord collapses to a const): every
            # candidate is a zero-extent relabel yielding the same layout, so the
            # first candidate is safe.
            return grow[0]

        size1_dim = _size1_alloc_dim(stl, tiebreak=_old_stick_size1_dim)
        if size1_dim is not None:
            op._size1_stick_alloc_dim = size1_dim
            logger.debug(
                "insert_restickify_padding: tagged size-1 output %s device dim %d "
                "for scheduler-window alloc grow",
                op.get_name(),
                size1_dim,
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
    if stl.device_size[device_dim] % get_elem_in_stick(out_layout.dtype) == 0:
        return

    old_dim_size = stl.device_size[device_dim]
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
    symbol to project onto a device dim, so fall back to the singleton device dim
    located by ``_size1_alloc_dim`` (declining if ambiguous).  This size-1 branch
    is now a defensive backstop: ``_pad_restickify_input`` declines a size-1
    new-stick read up front, so the restickify's own read no longer reaches here.
    """
    layout = producer.get_layout()
    write_dep = _named_write_dep(producer)
    host_coords = host_coordinates(layout, write_dep, None)
    stl = layout.device_layout
    sym = _single_free_sym(host_coords[new_stick_dim])
    if sym is not None:
        return _device_dim_carrying_sym(stl, write_dep, sym)
    # Size-1 host dim: locate the singleton producer device dim (declining if two
    # or more -1 markers make the choice ambiguous).
    return _size1_alloc_dim(stl)


def _grow_input_stick_dim(
    buf: ComputedBuffer, new_stick_dim: int, grow_host_dim: int | None
) -> bool:
    """Grow ``buf``'s device_size on the dim carrying ``new_stick_dim`` up to the
    stick boundary, so the restickify's stick-aligned over-read lands inside
    ``buf``'s own (wider) allocation instead of uninitialised HBM.

    The single read-side grow, shared by both input-padding entry points: the
    producer we own (``buf`` is the input) and the identity clone of a graph
    input (``buf`` is the clone).  Returns False if the geometry does not expose a
    bumpable device dim.  Both callers now treat False as unreachable and assert on
    it: a fresh contiguous clone always exposes the dim by construction, and a
    producer reaching this point has a single non-size-1 free symbol on a
    restickified dim (guaranteed by ``_pad_restickify_input``'s up-front declines),
    which ``_device_dim_carrying_sym`` must be able to place -- so a False from
    either is an invariant violation, not a case to handle.

    ``grow_host_dim`` selects the two cases, and it is load-bearing:

    - **Producer** (``grow_host_dim = new_stick_dim``): grow host_size too, so the
      producer's own iteration space computes the widened tail.  A Pointwise
      producer then writes real values into it; a Reduction (matmul) producer
      leaves it garbage.  Both are safe -- the tail never reaches a read position
      (every consumer iterates its logical extent and stops before it, the same
      bound that discards the restickify output's padded rows).
    - **Identity clone** (``grow_host_dim = None``): device_size only.  The clone
      is a pure copy that reads the *unpadded* graph input, so growing its host
      would make the clone itself over-read that input -- the very hazard we are
      fixing.  Its allocation is wider than its iteration; the restickify's
      over-read into the clone tail is don't-care.

    In both cases stride_map and host_stride are unchanged (see
    ``_pad_layout_device_dim``).
    """
    device_dim = _restickify_input_device_dim(buf, new_stick_dim)
    if device_dim is None:
        return False

    layout = buf.get_layout()
    old_dim_size = layout.device_layout.device_size[device_dim]
    n = concretize_expr(layout.size[new_stick_dim])
    new_dim_size = n + compute_padding(n, layout.dtype)

    buf.layout = _pad_layout_device_dim(
        layout, device_dim, new_dim_size, grow_host_dim=grow_host_dim
    )

    logger.debug(
        "insert_restickify_padding: fused pad into %s %s device dim %d %d -> %d "
        "(new stick host dim %d)",
        "producer" if grow_host_dim is not None else "clone",
        buf.get_name(),
        device_dim,
        old_dim_size,
        new_dim_size,
        new_stick_dim,
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


def _assert_input_paddable(
    op: ComputedBuffer, in_dep, in_layout, new_stick_dim: int
) -> None:
    """Raise ``Unsupported`` for restickify inputs the stick-boundary grow cannot
    pad, classifying each input dim's read by its coordinate.

    The grow widens the read to a stick boundary; it cannot re-base a read that
    begins partway into a stick (a stick has no start offset -- it always begins
    at its first element), nor gather non-adjacent rows.  Two read shapes are
    unpaddable and must fail loudly, not miscompile:

    - **Strided** read of any dim (coord ``k*var``, k not in {0, 1}: step > 1 or
      reversed), e.g. ``x[::2].transpose(1, 2).clone()``.  Codegen masks only a
      *contiguous* tail, so it would silently read the wrong rows.
    - **Narrowing slice on the new-stick dim** (iter range < dim size), e.g.
      ``x[:, :, 1:66, :].transpose(-2, -1).clone()``, whose slice becomes the new
      stick and starts partway into a stick.  Both input-pad paths grow the read
      feeding the restickify, so the displacement survives into codegen either way.

    TODO: both raises are removable by double-restickifying -- a re-base copy that
    reads the sliced/strided source and materializes a fresh buffer starting at a
    stick boundary, then restickifies that (the grow can pad such a buffer).

    Two read shapes are fine and flow through unchanged:

    - **Contiguous offset** on a non-stick dim (coord ``var + c``), e.g.
      ``x[:, 1:, :]``: codegen's offset/gap primitive carries it.
    - **Broadcast** read: a dim iterated wider than its size, so the same elements
      are re-read, e.g. ``k.view(B, S, H, D).transpose(1, 2).transpose(2, 3)`` on a
      ``[B, S, H, 2, 1, D/2]`` input, whose folded size-2 dim is read repeatedly.
      Its coordinate has a zero coefficient (``floor(v/64)`` / ``Mod(v, 64)`` --
      the loop var does not advance the read position), which is why it is not a
      stride: codegen takes the read's strides from the device layout rather than
      this coefficient, so the repeated block is read correctly.  The broadcast dim
      is never the dim we grow, so it is left as-is while the new-stick dim is
      padded (see ``test_broadcast_input_transpose_clone``).
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
        iter_range = _dep_range(in_dep, sym)
        dim_size = concretize_expr(in_layout.size[i])
        if iter_range is not None and iter_range != dim_size:
            raise Unsupported(
                f"insert_restickify_padding: sliced input on host dim "
                f"{i} of {op.get_name()} (iter range {iter_range} != "
                f"dim size {dim_size}) is not supported"
            )


def _new_stick_aliases_input_stick(
    op: ComputedBuffer, in_dep, in_layout, in_stick_dim: int
) -> bool:
    """Return True when the output stick is carved from the input's OWN stick and
    that input stick fills a whole stick -- the read-side skip condition.

    The over-read the input padding exists to cover stays inside already-
    initialized lanes here, so there is nothing to pad; growing the dim anyway
    would widen a device dim that may carry a tracked named dim (e.g. a matmul's
    D -- test_permute_matmul_distinct_lqlk).  Detected by projecting the output
    stick back through the input read: landing on ``in_stick_dim`` means it
    aliases the input stick.  A projection to None (sub-stick->sub-stick) or a
    different dim (a real transpose) is not an alias.

    The "inside already-initialized lanes" rationale only holds when the input
    stick fills a WHOLE stick: the projection matches by free *symbol*, so it is
    blind to the stick dim's size and reports the alias whether the input stick is
    a full 64 (D=64) or sub-stick (D=48).  When it is sub-stick the widened read
    runs past the initialized lanes into uninitialized HBM -- codegen still
    restickifies and reads garbage (a matmul with a sub-stick contraction dim
    miscompiles).  So only treat it as an alias when the input stick is itself
    stick-aligned; a sub-stick alias falls through to the grow.

    The "is the input stick full" check reads the within-stick symbol's iteration
    range straight off the read dep, not ``host_size[in_stick_dim]``: the loop var
    over the stick dim iterates its true (logical) fill -- 64 for a full D, 2/48
    for a sub-stick dim -- so ``range % stick_size`` is exactly the "already a
    whole stick" test.  This keys off the same within-stick symbol codegen uses and
    avoids re-indexing ``host_size`` by the projected dim, but the range and the
    host size of the stick dim are the same number (the device extent is
    stick-rounded and cannot distinguish full from sub-stick, which is why neither
    can be replaced by ``device_size``).
    """
    stick_size = get_elem_in_stick(in_layout.dtype)
    projected = _project_stick_host_dim(in_layout, op.get_layout(), in_dep)
    if projected is None or projected != in_stick_dim:
        return False
    in_dev_coords = _device_coords(in_layout.device_layout, in_dep)
    in_stick_sym = _single_free_sym(in_dev_coords[-1]) if in_dev_coords else None
    in_stick_fill = _dep_range(in_dep, in_stick_sym)
    if in_stick_fill is None:
        return False
    return in_stick_fill % stick_size == 0


def _pad_restickify_input(
    op: ComputedBuffer,
    operations: list[Operation],
    in_dep,
    in_buf,
    in_layout,
    new_stick_dim: int,
    in_stick_dim: int,
) -> None:
    """Read-side fix: ensure the restickify reads a grow-able ``ComputedBuffer``
    whose stick-carrying dim is padded to a stick boundary.

    Declines up front when there is no over-read to cover -- the new stick dim is
    already a stick multiple (so the read never runs past the true dim size), it
    is a size-1 host dim (one real lane, nothing past it; codegen's
    ``_restore_elided_restickify_stick`` supplies the padded lanes on the elided
    output stick), or it is carved from the input's OWN aligned stick
    (``_new_stick_aliases_input_stick``): the widened read stays inside
    already-initialized lanes, so there is nothing to pad and growing the dim
    would only widen a device dim that may carry a tracked named dim.

    The candidate splits into two mutually-exclusive arms on whether we own the
    input buffer; they share the same grow (``_grow_input_stick_dim``) but differ
    in what they grow and whether growing can fail:

    - **Producer arm** -- the input is a ``ComputedBuffer`` we produced.  Grow it
      in place (cheap: no extra buffer, no copy, no HBM round-trip).  The grow
      cannot fail here: after the size-1 decline above, ``new_stick_dim`` carries a
      single non-size-1 free symbol, and ``is_restickify`` fired for this candidate
      (in-stick != out-stick), so that symbol must project onto a device dim other
      than the within-stick one.  A False return is an invariant violation and
      raises -- it never silently escalates to the clone arm.
    - **Clone arm** -- the input is a graph input whose allocation is not ours to
      widen (not a ``ComputedBuffer``).  Insert an identity clone ahead of the
      restickify, grow the clone device-size-only, and redirect the restickify to
      read it.  A fresh contiguous clone always exposes the device dim, so its grow
      is asserted too.

    The clone's grow is device_size-only (``grow_host_dim=None``) while the
    producer's also grows host_size -- see ``_grow_input_stick_dim`` for why.

    A read the grow cannot carry (mid-stick start on the new-stick dim, or a
    strided read) is refused loudly (``_assert_input_paddable``) up front, before
    either arm: the read feeding the restickify is the same regardless of which
    buffer we grow, so growing the producer we own does not undo it any more than
    growing a clone would.  The producer arm is grown in place with no separate
    read, so the guard is the only thing standing between it and a silent
    miscompile.
    """
    # Skip when the new-stick DIM is already a stick multiple: the read never runs
    # past the true dim size, so there is nothing to pad.  This keys off the dim's
    # declared size, not the read's iteration range -- for a slice that lands on a
    # dim the transpose turns non-stick, the range is the narrowed extent while the
    # dim is a full stick (x[3:66].transpose(0, 1) -> new_stick_dim size 128, range
    # 63); the aligned full dim must skip, and the narrowing is a carried offset a
    # non-stick dim tolerates (test_sliced_transpose_stick_expr_compiles).
    host_size = [concretize_expr(s) for s in in_layout.size]
    if compute_padding(host_size[new_stick_dim], in_layout.dtype) == 0:
        return
    # Skip when the new-stick dim is a size-1 host dim (symbol-free coord): there
    # is no over-read to cover.  The read is fully described by the input's live
    # OLD stick (already padded to a stick boundary); the new stick carries one
    # real lane and no data past it, so there is nothing to pad.  Codegen's
    # _restore_elided_restickify_stick inserts the 63 padded lanes on the elided
    # output stick, and none of them is ever read.  Growing here would only widen
    # a device dim that may carry a tracked named dim, and the size-1 device dim
    # is the -1-marked singleton the codegen restore handles directly.
    in_host_coords = host_coordinates(in_layout, in_dep, None)
    if _single_free_sym(in_host_coords[new_stick_dim]) is None:
        return
    if _new_stick_aliases_input_stick(op, in_dep, in_layout, in_stick_dim):
        return

    # Guard both branches: the offending read is the restickify's, not the buffer
    # we grow, so it defeats the producer-grow path just as it defeats the clone
    # path.
    _assert_input_paddable(op, in_dep, in_layout, new_stick_dim)

    # Producer arm: a ComputedBuffer we produced.  Grow it in place (cheap: no
    # extra buffer, no copy, no HBM round-trip).  The grow must succeed here --
    # after the size-1 decline above, new_stick_dim carries a single non-size-1
    # free symbol, and is_restickify fired for this candidate (in-stick !=
    # out-stick), so _device_dim_carrying_sym finds a device dim other than the
    # within-stick one.  A False is a can't-happen invariant violation, not a case
    # to silently escalate to the clone arm.
    if isinstance(in_buf, ComputedBuffer):
        grew = _grow_input_stick_dim(in_buf, new_stick_dim, grow_host_dim=new_stick_dim)
        assert grew, (
            f"_pad_restickify_input: producer ComputedBuffer {in_buf.get_name()} "
            f"exposed no bumpable device dim for new stick dim {new_stick_dim} -- "
            f"after the size-1 decline this is a single non-size-1 free symbol on a "
            f"restickified dim, which must project onto a device dim other than the "
            f"within-stick one; a False here is a can't-happen invariant violation, "
            f"not a case to clone-escalate."
        )
        return

    # Clone arm (graph input only): in_buf is not a ComputedBuffer, so its
    # allocation is not ours to widen.  Materialise an identity clone ahead of the
    # restickify, grow the clone device-size-only, and redirect the read to it.
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
    # device-size-only (the clone copies just the real rows).
    grew = _grow_input_stick_dim(clone_buf, new_stick_dim, grow_host_dim=None)
    assert grew, (
        f"_pad_restickify_input: no device dim carrying new stick dim "
        f"{new_stick_dim} for clone {clone_buf.get_name()}"
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
    - Read side (``_pad_restickify_input``): when the new stick dim's size is not
      a stick multiple the read runs past the true dim size.  Grows the producer
      in place when we own the buffer (a ``ComputedBuffer``); for a graph input we
      do not own, inserts a device-size-bumped identity clone and redirects the
      restickify to read it.

    The two are orthogonal — e.g. a 128x67 transpose has an aligned input
    stick dim (128) but an unaligned non-stick dim (67), so only the write-side
    fix fires.
    """
    operations = graph.operations
    for op in list(operations):
        match = _identify_restickify_candidate(op, graph)
        if match is None:
            continue
        in_dep, in_buf, in_layout, new_stick_dim, in_stick_dim = match
        # ComputedBuffer guaranteed by _identify_restickify_candidate
        assert isinstance(op, ComputedBuffer)

        _pad_restickify_output(op, in_dep, in_layout, in_stick_dim)
        _pad_restickify_input(
            op, operations, in_dep, in_buf, in_layout, new_stick_dim, in_stick_dim
        )
