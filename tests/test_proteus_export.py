"""Check both released actor interfaces through a small deterministic ONNX graph."""
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from scripts.export_proteus_onnx import compare_with_source, input_shapes, source_inputs, wrap_actor


def contract(kind):
    return dict(actor_kind=kind, robot="unitree_g1", history=3, seq_window=10,
                num_slots=2, body_names=["pelvis"], future_offsets=[0, -1],
                far_offset_range=[5, 33], morph=[.25, -.5], joints=[{"slot": 1}])


def actor(c):
    shapes = input_shapes(c, 16)
    inputs = []
    nodes, initializers, columns = [], [], []
    named = {old: shapes[new] for old, new in source_inputs(c).items()} | {"morph": (2,)}
    initializers.append(numpy_helper.from_array(np.array([1], np.int64), "column_axis"))
    for idx, (name, shape) in enumerate(named.items()):
        inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", *shape]))
        axes = f"sum_axes_{idx}"
        initializers.append(numpy_helper.from_array(np.arange(1, len(shape)+1, dtype=np.int64), axes))
        column = f"column_{idx}"
        nodes.extend([helper.make_node("ReduceSum", [name, axes], [f"sum_{idx}"], keepdims=0),
                      helper.make_node("Unsqueeze", [f"sum_{idx}", "column_axis"], [column])])
        columns.append(column)
    nodes.append(helper.make_node("Sum", columns, ["total"]))
    nodes.append(helper.make_node("Concat", ["total", "total"], ["actions"], axis=1))
    graph = helper.make_graph(nodes, "actor", inputs,
                              [helper.make_tensor_value_info("actions", TensorProto.FLOAT, ["batch", 2])], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    return model


@pytest.mark.parametrize("kind", ["separate", "wam_seq"])
def test_wrapper_preserves_actor_inputs_morph_and_driven_slots(tmp_path, kind):
    c = contract(kind)
    source = actor(c)
    path = tmp_path / "actor.onnx"
    onnx.save(source, path)
    wrapped = wrap_actor(source, c, 16)
    assert wrapped.ir_version == 8
    assert {i.name for i in wrapped.graph.input} == set(input_shapes(c, 16))
    assert compare_with_source(wrapped, path, c, 16, 24, 123) <= 1e-6
    if kind == "wam_seq":
        assert input_shapes(c, 16)["proprioception"][0] == 10
        assert c["history"] == 3


def test_wam_contract_rejects_legacy_actor():
    with pytest.raises(ValueError, match="inputs"):
        wrap_actor(actor(contract("separate")), contract("wam_seq"), 16)


def test_nonfinite_actions_cannot_pass_export_comparison(tmp_path):
    c = contract("wam_seq")
    c["morph"][0] = np.nan
    source = actor(c)
    path = tmp_path / "actor.onnx"
    onnx.save(source, path)
    wrapped = wrap_actor(source, c, 16)
    with pytest.raises(RuntimeError, match="non-finite"):
        compare_with_source(wrapped, path, c, 16, 1, 123)
