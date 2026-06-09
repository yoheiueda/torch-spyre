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

import logging
from collections import defaultdict

import torch

from .constants import ELIDED_COPY_BACK_ATTR
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger
from .pass_utils import copy_fx_custom_meta
from torch._inductor.ir import (
    ComputedBuffer,
    FixedLayout,
    InputBuffer,
    MutationLayoutSHOULDREMOVE,
    Operation,
    ReinterpretView,
    StorageBox,
    TensorBox,
)
from torch_spyre._C import SpyreTensorLayout
from torch._inductor.ops_handler import WrapperHandler
from torch._inductor.virtualized import V

from torch.utils._ordered_set import OrderedSet


logger = get_inductor_logger("insert_restickify")


def _fixed_tiled(layout: FixedLayout, stl: SpyreTensorLayout) -> FixedTiledLayout:
    return FixedTiledLayout(
        layout.device, layout.dtype, layout.size, layout.stride, stl
    )


def _record_restickify(
    op: Operation,
    dep_name: str,
    target_layout: FixedTiledLayout,
    restickify_plan: dict,
) -> None:
    """Record that op's input arg_name must be restickified to target_layout.

    restickify_plan is the deferred execution queue: entries are recorded here during
    finalize_layouts and executed later by insert_restickify.
    """
    restickify_plan[op.get_name()].append(
        {"arg_name": dep_name, "target_layout": target_layout}
    )


class NameSwapHandler(WrapperHandler):
    """
    Wrapper to patch a node's inner_fn to use new buffer names after inserting
    nodes upstream that change the input buffers.
    """

    def __init__(self, inner, name_map: dict[str, str]):
        super().__init__(inner)
        self._name_map = name_map

    def load(self, name, index):
        return super().load(self._name_map.get(name, name), index)

    def store(self, name, index, value, mode=None):
        return super().store(self._name_map.get(name, name), index, value, mode)


def _create_restickify_node(
    restick_arg_info: dict, op: ComputedBuffer
) -> tuple[str, ComputedBuffer]:
    """
    Lower a restickify FX node for the given incompatible input arg.

    Inserts a spyre.restickify call into the FX graph, lowers it via
    graph_lowering.run_node(), and assigns the target layout.  Returns
    (old_buffer_name, new_computed_buffer).
    """
    arg_name = restick_arg_info["arg_name"]

    graph_lowering = V.graph
    fx_graph = graph_lowering.graph

    # View ops (e.g. permute) lower to ReinterpretView with no buffer name and
    # are absent from env. Patch env from name_to_users so the search below can
    # resolve them.
    env = {}
    for tbs in graph_lowering.name_to_users.values():
        for tb in tbs:
            if not tb.data.origins:
                continue
            tb_fx_node = list(tb.data.origins)[0]
            env[tb_fx_node] = tb
    graph_lowering.env.update(env)

    # Search env by buffer name to find the FX node to pass to restickify.
    fx_arg_node = next(
        fx_node
        for fx_node, tb in graph_lowering.env.items()
        if isinstance(fx_node, torch.fx.Node)
        and isinstance(tb, TensorBox)
        and tb.get_name() == arg_name
    )
    # Insert at a valid position in the FX graph; the operations list order is
    # authoritative pre-scheduler, not position in the FX graph.
    first_compute_node = next(n for n in fx_graph.nodes if n.op != "placeholder")
    with fx_graph.inserting_before(first_compute_node):
        restick_fx_node = fx_graph.create_node(
            "call_function", torch.ops.spyre.restickify.default, (fx_arg_node,)
        )
    # Propagate hint metadata from the consumer op so assign_dim_hints can assign
    # dim_hints to the restickify buffer after insertion.
    for consumer_fx_node in op.origins:
        if "custom" in consumer_fx_node.meta:
            copy_fx_custom_meta(consumer_fx_node, restick_fx_node)
            break
    # Lower the FX node; run_node registers the output in graph.buffers and graph.operations.
    restick_tb = graph_lowering.run_node(restick_fx_node)
    restick_buff = restick_tb.data.data  # TensorBox -> StorageBox -> ComputedBuffer
    assert isinstance(restick_buff, ComputedBuffer), (
        f"Expected ComputedBuffer, got {type(restick_buff).__name__}"
    )
    # origins is empty by default since spyre.restickify has no ATen decomposition;
    # set it to the synthetic FX node so code that expects non-empty origins doesn't crash.
    restick_buff.origins = OrderedSet([restick_fx_node])
    graph_lowering.env[restick_fx_node] = restick_tb

    restick_buff.layout = restick_arg_info["target_layout"]

    return arg_name, restick_buff


def insert_restickify_on_node_inputs(
    op: ComputedBuffer,
    resticks_needed: list[dict],
    operations: list[Operation],
) -> None:
    """Insert restickify nodes before op for each incompatible input, patch op's inner_fn
    to read the new buffer names, and reconstruct the consumer ComputedBuffer to
    invalidate its sizes cache.
    """
    name_map = {}
    try:
        op_index = operations.index(op)
    except ValueError:
        raise AssertionError(
            f"Consumer op {op.get_name()} not found in operations list"
        ) from None

    for restick_arg_info in resticks_needed:
        old_name, restick_buff = _create_restickify_node(restick_arg_info, op)
        name_map[old_name] = restick_buff.get_name()

        # lower_restickify calls pw.realize() which appends restick_buff to operations.
        # Move it to just before the consumer op to preserve topological order.
        operations.remove(restick_buff)
        operations.insert(op_index, restick_buff)
        op_index += 1  # consumer shifted right by 1

    # Patch inner_fn once with the full name_map covering all restickified args.
    orig_inner = op.data.inner_fn

    def new_inner_fn(*args, _map=name_map, _orig_inner=orig_inner):
        with V.set_ops_handler(NameSwapHandler(V.ops, _map)):
            return _orig_inner(*args)

    object.__setattr__(op.data, "inner_fn", new_inner_fn)

    # Reconstruct ComputedBuffer as a fresh object so the instance-keyed cache
    # on get_default_sizes_body can be cleanly invalidated below.
    new_consumer_buffer = ComputedBuffer(
        name=op.get_name(),
        layout=op.layout,
        data=op.data,
        _split_size=op._split_size,
        _original_inner_fn=op._original_inner_fn,
        _original_ranges=op._original_ranges,
        _original_reduction_ranges=op._original_reduction_ranges,
    )
    new_consumer_buffer.operation_name = op.operation_name
    new_consumer_buffer.origins = op.origins
    # Replace op in the operations list with the reconstructed buffer.
    operations[op_index] = new_consumer_buffer
    V.graph.name_to_buffer[new_consumer_buffer.get_name()] = new_consumer_buffer

    # Invalidate the sizes/body cache so it is recomputed on next access with the patched inner_fn.
    ComputedBuffer.get_default_sizes_body.clear_cache(new_consumer_buffer)


def insert_restickify(operations: list[Operation]) -> None:
    """Insert restickify operations before all nodes in restickify_plan.

    Consumes V.graph.restickify_plan (built by finalize_layouts) and splices the
    necessary ComputedBuffer nodes into the operations list in-place.
    No scheduler state is touched.
    """
    restickify_plan = V.graph.restickify_plan
    if not restickify_plan:
        return

    for op in list(
        operations
    ):  # copy since insert_restickify_on_node_inputs mutates operations
        if isinstance(op, ComputedBuffer) and op.get_name() in restickify_plan:
            insert_restickify_on_node_inputs(
                op, restickify_plan[op.get_name()], operations
            )


def finalize_layouts(operations: list) -> None:
    """Convert committed STLs (set by the optimizer) to FixedTiledLayouts and build
    V.graph.restickify_plan for insert_restickify.

    Two steps:
    - Commit: wrap each op's committed_stl in a FixedTiledLayout and assign it to
      op.layout; clean up optimizer-only attributes (layouts, restick_cost_fn,
      committed_stl).
    - Schedule restickifies: for each input edge where the committed input STL is
      incompatible with what the op requires, record a restickify in the plan.
    """
    for name in V.graph.graph_input_names:
        tensor_box = V.graph.graph_inputs[name]
        if (
            isinstance(tensor_box, TensorBox)
            and isinstance(tensor_box.data, StorageBox)
            and isinstance(tensor_box.data.data, InputBuffer)
            and hasattr(tensor_box, "layouts")
        ):
            input_buf = tensor_box.data.data
            assert hasattr(input_buf, "committed_stl"), (
                f"graph input {name} has no committed_stl — optimizer did not run"
            )
            stl = input_buf.committed_stl
            input_buf.layout = _fixed_tiled(input_buf.layout, stl)
            del tensor_box.layouts

    plan: dict = defaultdict(list)

    for op in operations:
        cost_fn = getattr(op, "restick_cost_fn", None)
        op_layouts = getattr(op, "layouts", None)
        committed = getattr(op, "committed_stl", None)
        for attr in ("layouts", "restick_cost_fn", "committed_stl"):
            if hasattr(op, attr):
                delattr(op, attr)

        # Commit the chosen STL and wrap in a FixedTiledLayout
        # Exclude mutation ops because their op.layout must not be set
        # until after the scheduler runs
        if op_layouts and not isinstance(op.layout, MutationLayoutSHOULDREMOVE):
            stl = committed if cost_fn else op_layouts[0]
            op.layout = _fixed_tiled(op.layout, stl)

        # For each input edge, schedule a restickify if the input's committed STL
        # is incompatible with what this op requires on that edge.
        if not cost_fn:
            continue
        for edge, target_stl in cost_fn.required_input_stls(committed):
            input_buf = V.graph.get_buffer(edge.dep.name)
            in_layout = input_buf.get_layout()
            if isinstance(in_layout, MutationLayoutSHOULDREMOVE):
                assert getattr(input_buf, ELIDED_COPY_BACK_ATTR, False), (
                    f"unexpected mutation layout on {edge.dep.name}"
                )
                in_layout = in_layout.real_layout()
            in_stl = in_layout.device_layout
            restick_stl = edge.layout(in_stl, target_stl)
            if restick_stl is None:
                continue
            restick_target = _fixed_tiled(in_layout, restick_stl)
            logger.info(
                f"Injecting restickify on {op.get_name()} input {edge.dep.name}: "
                f"{list(in_stl.stride_map)} -> {list(target_stl.stride_map)}"
            )
            _record_restickify(op, edge.dep.name, restick_target, plan)

    V.graph.restickify_plan = plan
    if logger.isEnabledFor(logging.DEBUG):
        if plan:
            lines = ["restickify plan:"]
            for op_name, resticks in plan.items():
                consumer = V.graph.get_buffer(op_name)
                if isinstance(consumer, ComputedBuffer) and hasattr(
                    consumer.data, "reduction_type"
                ):
                    op_kind = f"reduction:{consumer.data.reduction_type}"
                elif isinstance(consumer, ComputedBuffer):
                    op_kind = "pointwise"
                else:
                    op_kind = type(consumer).__name__
                for r in resticks:
                    tgt = r["target_layout"]
                    arg_name = r["arg_name"]
                    arg_buf = V.graph.get_buffer(arg_name)
                    if (
                        isinstance(arg_buf, TensorBox)
                        and isinstance(arg_buf.data, StorageBox)
                        and isinstance(arg_buf.data.data, InputBuffer)
                    ):
                        buf_kind = "graph_input"
                    elif isinstance(arg_buf, ComputedBuffer):
                        buf_kind = "computed"
                    else:
                        buf_kind = type(arg_buf).__name__
                    lines.append(
                        f"  restickify {arg_name} ({buf_kind}) -> {op_name} ({op_kind})"
                        f"  stride_map={list(tgt.device_layout.stride_map)}"
                    )
            logger.debug("\n".join(lines))
        else:
            logger.debug("restickify plan: (none)")


def _find_mutation_op(
    operations: list[Operation], target_name: str
) -> "ComputedBuffer | None":
    """Find the mutation ComputedBuffer whose peeled target name matches target_name."""
    for op in operations:
        if not isinstance(op, ComputedBuffer):
            continue
        if not isinstance(op.layout, MutationLayoutSHOULDREMOVE):
            continue
        tgt = op.layout.target
        while isinstance(tgt, ReinterpretView):
            tgt = tgt.data
        if hasattr(tgt, "get_name") and tgt.get_name() == target_name:
            return op
    return None


def insert_post_mutation_restickify(operations: list[Operation]) -> None:
    """Insert pre/post kernels around a slice mutation to handle unsupported stick offsets.

    When the mutation target's orig_stl produces an unsupported stick expression for
    the slice write, propagate_layouts selects an alt_stl and records the pair in
    V.graph.mutation_restick_needed.  This pass inserts three kernels:

      1. pre-restickify:  arg0_1 (orig_stl) → buf_tmp (alt_stl)   [layout-change copy]
      2. mutation:        pool_ones → buf_tmp[slice] (alt_stl)     [slice write]
      3. copy-back:       buf_tmp (alt_stl) → arg0_1 (alt_stl)    [identity copy]
      4. set_spyre_tensor_layout(arg0_1, alt_stl)                  [DCI metadata fix]

    Store redirects for the mutation and copy-back ops are recorded in
    V.graph._store_redirects and applied at codegen time in SpyreKernel.store().
    NameSwapHandler cannot be used here because for a ComputedBuffer the store target
    name is passed to SpyreKernel.store() from outside inner_fn — inner_fn only
    produces the value (loader); the NameSwapHandler wrapping V.ops only intercepts
    ops.load/ops.store calls made *inside* inner_fn.

    arg0_1 is returned unchanged so DCI downloads from its stable HBM address.
    """
    plan = getattr(V.graph, "mutation_restick_needed", {})
    if not plan:
        return

    graph_lowering = V.graph
    if not hasattr(graph_lowering, "_store_redirects"):
        graph_lowering._store_redirects = {}
    if not hasattr(graph_lowering, "_pre_restickify_input_stl"):
        graph_lowering._pre_restickify_input_stl = {}

    for target_name, (orig_stl, alt_stl) in plan.items():
        mutation_op = _find_mutation_op(operations, target_name)
        if mutation_op is None:
            logger.warning(
                "insert_post_mutation_restickify: no mutation op found for %s",
                target_name,
            )
            continue

        # Retrieve the InputBuffer for target_name (after finalize_layouts, layout=alt_stl).
        target_tb = graph_lowering.graph_inputs.get(target_name)
        if target_tb is None:
            logger.warning(
                "insert_post_mutation_restickify: %s not in graph_inputs", target_name
            )
            continue
        target_input_buf = target_tb.data.data  # TensorBox → StorageBox → InputBuffer
        base_layout = target_input_buf.layout  # FixedTiledLayout(alt_stl)
        # Create fresh layouts with empty allocation dicts so memory_planning assigns
        # independent pool/HBM addresses.  Reusing base_layout directly would share its
        # allocation dict and cause memory_planning to stomp arg0_1's HBM address.
        buf_tmp_layout = _fixed_tiled(base_layout, alt_stl)
        buf_copyback_layout = _fixed_tiled(base_layout, alt_stl)

        # Step 1: create pre-restickify node: arg0_1 (orig_stl) → buf_tmp (alt_stl).
        # Override so create_tensor_arg reads arg0_1 with orig_stl (not the committed alt_stl).
        orig_stl_layout = _fixed_tiled(base_layout, orig_stl)
        graph_lowering._pre_restickify_input_stl[target_name] = orig_stl_layout
        _, buf_tmp = _create_restickify_node(
            {"arg_name": target_name, "target_layout": buf_tmp_layout},
            mutation_op,
        )
        buf_tmp_name = buf_tmp.get_name()

        # Step 2: redirect mutation's store to buf_tmp instead of its own buffer name.
        # The mutation op's own buffer name becomes a dead allocation; add to removed_buffers
        # so the scheduler/codegen does not try to allocate or pass it as a kernel arg.
        mutation_name = mutation_op.get_name()
        graph_lowering._store_redirects[mutation_name] = buf_tmp_name
        graph_lowering.removed_buffers.add(mutation_name)

        # Step 3: create copy-back node: buf_tmp (alt_stl) → buf_copyback (alt_stl).
        # Redirect its store to target_name (arg0_1) so the kernel writes the full
        # restickified tensor back to the original HBM address.
        # MutationLayoutSHOULDREMOVE makes the scheduler treat copy-back as a live
        # mutation of arg0_1, preventing DCE of buf_copyback (and transitively buf_tmp).
        _, buf_copyback = _create_restickify_node(
            {"arg_name": buf_tmp_name, "target_layout": buf_copyback_layout},
            mutation_op,
        )
        copyback_name = buf_copyback.get_name()
        buf_copyback.layout = MutationLayoutSHOULDREMOVE(target_tb)
        graph_lowering._store_redirects[copyback_name] = target_name

        # Insert buf_tmp before mutation, copy-back after mutation.
        mutation_op_index = operations.index(mutation_op)
        operations.remove(buf_tmp)
        operations.insert(mutation_op_index, buf_tmp)
        # mutation_op is now at mutation_op_index + 1; insert copy-back after it.
        operations.remove(buf_copyback)
        operations.insert(mutation_op_index + 2, buf_copyback)

        logger.info(
            "insert_post_mutation_restickify: "
            "%s (orig→alt) before %s; copy-back %s→%s after %s",
            target_name,
            mutation_name,
            buf_tmp_name,
            target_name,
            mutation_name,
        )
