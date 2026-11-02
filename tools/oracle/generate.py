#!/usr/bin/env python3
"""Generate golden parity fixtures for JTorch from a pinned PyTorch runtime.

Run from the repository root:

    python3 tools/oracle/generate.py

The fixtures are checked in under tests/parity/src/test/resources/golden. Each fixture describes the operator
arguments in schema order, the expected outputs (shape, strides, storage offset, dtype, values) and the gradients
of ``sum(floating outputs)`` with respect to differentiable tensor arguments. The Java parity runner reconstructs
the call generically from the fixture, so no per-operator Java test code is hand-maintained.

Random and uninitialised factories are recorded as metadata-only fixtures: JTorch deliberately does not promise
bit-identical random streams with PyTorch, only statistical and seed-reproducibility semantics.
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/parity/src/test/resources/golden"

DTYPE_NAMES = {
    torch.float64: "float64",
    torch.float32: "float32",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.bool: "bool",
}

torch.set_default_dtype(torch.float64)


def tensor(values, dtype="float64", requires_grad=False, shape=None):
    dtype_map = {
        "float64": torch.float64,
        "float32": torch.float32,
        "int64": torch.int64,
        "int32": torch.int32,
        "bool": torch.bool,
    }
    t = torch.tensor(values, dtype=dtype_map[dtype])
    if shape is not None:
        t = t.reshape(shape)
    if requires_grad:
        if not t.is_floating_point():
            raise ValueError("only floating tensors can require grad")
        t.requires_grad_(True)
    return t


def flatten(t):
    return t.reshape(-1).tolist()


def encode_tensor(t, include_data=True):
    return {
        "kind": "tensor",
        "dtype": DTYPE_NAMES[t.dtype],
        "shape": list(t.shape),
        "strides": list(t.stride()),
        "storageOffset": t.storage_offset(),
        "requiresGrad": bool(t.requires_grad),
        "data": flatten(t) if include_data else [],
    }


def encode_value(value):
    if value is None:
        return {"kind": "none"}
    if isinstance(value, torch.Tensor):
        return encode_tensor(value)
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": value}
    if isinstance(value, float):
        return {"kind": "float", "value": value}
    if isinstance(value, str):
        return {"kind": "string", "value": value}
    if isinstance(value, torch.dtype):
        return {"kind": "dtype", "value": DTYPE_NAMES[value]}
    if isinstance(value, torch.device):
        return {"kind": "device", "value": str(value)}
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(v, torch.Tensor) for v in value):
            return {"kind": "tensorlist", "value": [encode_tensor(v) for v in value]}
        return {"kind": "list", "value": [int(v) for v in value]}
    raise TypeError(f"cannot encode {value!r}")


CASES = []


def case(operator, args, nondeterministic=False, inplace=False, grads=True, note=None):
    CASES.append(
        {
            "operator": operator,
            "args": args,
            "nondeterministic": nondeterministic,
            "inplace": inplace,
            "grads": grads,
            "note": note,
        }
    )


def make_tensor_arg(value, dtype="float64", requires_grad=False, shape=None):
    return tensor(value, dtype=dtype, requires_grad=requires_grad, shape=shape)


def leaf(values, shape=None, dtype="float64"):
    t = tensor(values, dtype=dtype, shape=shape)
    t.requires_grad_(True)
    return t


def build_cases():
    def A():
        return leaf([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3))

    def B():
        return leaf([0.5, 1.5], (2, 1))

    def C():
        return leaf([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], (3, 4))

    def Positive6():
        return leaf([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], (2, 3))

    def NonContiguous():
        base = make_tensor_arg([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(2, 3)
        view = base.t()
        view.requires_grad_(True)
        return view

    # ------------------------------------------------------------------ creation
    case("aten::zeros.default", [[2, 3], torch.float64, torch.device("cpu")], grads=False)
    case("aten::ones.default", [[2, 3], torch.float64, torch.device("cpu")], grads=False)
    case("aten::empty.default", [[2, 3], torch.float64, torch.device("cpu")], nondeterministic=True, grads=False)
    case("aten::full.default", [[2, 2], 3.5, torch.float64, torch.device("cpu")], grads=False)
    case("aten::arange.default", [0, 5, 1, torch.float64, torch.device("cpu")], grads=False)
    case("aten::arange.default", [5, 0, -2, torch.int64, torch.device("cpu")], grads=False)
    case("aten::randn.default", [[2, 3], torch.float64, torch.device("cpu")], nondeterministic=True, grads=False)
    case("aten::rand.default", [[2, 3], torch.float64, torch.device("cpu")], nondeterministic=True, grads=False)
    case("aten::randint.default", [2, 10, [2, 3], torch.int64, torch.device("cpu")], nondeterministic=True, grads=False)
    case("aten::eye.default", [3, torch.float64, torch.device("cpu")], grads=False)

    # ------------------------------------------------------------------ views
    case("aten::reshape.default", [A(), [3, 2]])
    case("aten::view.default", [A(), [3, 2]])
    case("aten::reshape.default", [NonContiguous(), [6]], note="non-contiguous reshape copies")
    case("aten::transpose.default", [A(), 0, 1])
    case("aten::permute.default", [A(), [0, 1]])
    case("aten::narrow.default", [A(), 1, 1, 2])
    case("aten::select.default", [A(), 0, 1])
    squeezed = leaf([1.0, 2.0], (1, 2, 1))
    case("aten::squeeze.default", [squeezed])
    squeezed = leaf([1.0, 2.0], (1, 2, 1))
    case("aten::squeeze.dim", [squeezed, 0])
    case("aten::unsqueeze.default", [A(), 1])
    expanded = leaf([1.0, 2.0], (2, 1))
    case("aten::expand.default", [expanded, [2, 3]])
    case("aten::flatten.default", [A(), 0, 1])
    case("aten::t.default", [A()])
    case("aten::contiguous.default", [A()])
    case("aten::contiguous.default", [NonContiguous()], note="non-contiguous input copies")
    case("aten::clone.default", [A()])
    case("aten::detach.default", [A()], grads=False)
    case("aten::to.dtype", [A(), torch.float32])

    # ------------------------------------------------------------------ elementwise
    case("aten::add.Tensor", [A(), B(), 1])
    case("aten::add.Tensor", [A(), B(), 2.5])
    case("aten::sub.Tensor", [A(), B(), 0.5])
    case("aten::mul.Tensor", [A(), B()])
    case("aten::div.Tensor", [A(), B()])
    case("aten::pow.Tensor_Scalar", [A(), 2.0])
    positive = leaf([0.5, 1.5], (2, 1))
    case("aten::pow.Tensor_Tensor", [A(), positive])
    case("aten::maximum.default", [A(), B()])
    case("aten::minimum.default", [A(), B()])
    for name in ("eq", "ne", "lt", "le", "gt", "ge"):
        case(f"aten::{name}.Tensor", [A(), B()], grads=False)
    bools = make_tensor_arg([[True, False, True], [False, False, True]], dtype="bool")
    case("aten::logical_not.default", [bools], grads=False)

    for name in ("neg", "abs", "square", "sqrt", "exp", "log", "tanh", "sigmoid", "relu",
                 "reciprocal", "sin", "cos", "log1p"):
        case(f"aten::{name}.default", [Positive6()])
    for name in ("floor", "ceil", "sign"):
        case(f"aten::{name}.default", [Positive6()], grads=False)
    clamp_input = leaf([-2.0, -0.5, 0.0, 0.5, 1.5, 3.0], (2, 3))
    case("aten::clamp.default", [clamp_input, -1.0, 1.0])
    clamp_input = leaf([-2.0, -0.5, 0.0, 0.5, 1.5, 3.0], (2, 3))
    case("aten::clamp.default", [clamp_input, None, 0.5])

    # ------------------------------------------------------------------ reductions
    case("aten::sum.default", [A()])
    case("aten::sum.dim_IntList", [A(), [0], False])
    case("aten::sum.dim_IntList", [A(), [1], True])
    case("aten::mean.default", [A()])
    case("aten::mean.dim", [A(), [1], False])
    case("aten::prod.default", [make_tensor_arg([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(2, 3)], grads=False)
    case("aten::prod.dim", [make_tensor_arg([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(2, 3), 1, False],
         grads=False)
    case("aten::max.default", [A()])
    case("aten::min.default", [A()])
    case("aten::max.dim", [A(), 1, False])
    case("aten::min.dim", [A(), 0, True])
    case("aten::argmax.default", [A(), None, False], grads=False)
    case("aten::argmax.default", [A(), 1, False], grads=False)
    case("aten::argmin.default", [A(), None, False], grads=False)
    case("aten::argmin.default", [A(), 1, True], grads=False)
    case("aten::softmax.default", [A(), 1])
    case("aten::log_softmax.default", [A(), 0])

    # ------------------------------------------------------------------ matrix
    case("aten::matmul.default", [A(), C()])
    case("aten::mm.default", [A(), C()])
    batch_a = leaf(list(range(1, 13)), (2, 2, 3))
    batch_b = leaf([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5], (2, 3, 2))
    case("aten::bmm.default", [batch_a, batch_b])
    self_addmm = leaf([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], (2, 4))
    mat1 = leaf([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3))
    case("aten::addmm.default", [self_addmm, mat1, C(), 0.5, 2.0])
    weight = leaf([0.5, -0.5, 1.0, 0.25, 0.75, -0.25, 0.1, 0.2, 0.3, -0.1, -0.2, -0.3], (4, 3))
    bias = leaf([0.5, -0.5, 0.25, 0.75])
    case("aten::linear.default", [A(), weight, bias])

    # ------------------------------------------------------------------ convolution and pooling
    def ramp(n):
        return [i / 10.0 for i in range(n)]

    # Convolution accumulation order differs across CPU implementations (oneDNN/BLAS). Use dyadic inputs and
    # weights so every product and partial sum is exactly representable in float64; otherwise the committed
    # fixture can drift by one ULP between the development host and CI.
    def dyadic_ramp(n):
        return [i / 8.0 for i in range(n)]

    conv_input = leaf(dyadic_ramp(32), (1, 2, 4, 4))
    conv_weight = leaf(dyadic_ramp(54), (3, 2, 3, 3))
    conv_bias = leaf([0.125, -0.25, 0.375])
    case("aten::conv2d.default", [conv_input, conv_weight, conv_bias, [1, 1], [0, 0], [1, 1], 1])
    conv_input = leaf(dyadic_ramp(50), (1, 2, 5, 5))
    conv_weight = leaf(dyadic_ramp(36), (2, 2, 3, 3))
    case("aten::conv2d.default", [conv_input, conv_weight, None, [2, 2], [1, 1], [1, 1], 1])
    conv_input = leaf(dyadic_ramp(9), (1, 1, 3, 3))
    conv_weight = leaf(dyadic_ramp(8), (2, 1, 2, 2))
    case("aten::conv2d.default", [conv_input, conv_weight, None, [1, 1], [1, 1], [2, 2], 1])
    pool_input = leaf(ramp(32), (1, 2, 4, 4))
    case("aten::max_pool2d.default", [pool_input, [2, 2], [2, 2], [0, 0], [1, 1], False])
    pool_input = leaf(ramp(50), (1, 2, 5, 5))
    case("aten::max_pool2d.default", [pool_input, [3, 3], [2, 2], [1, 1], [1, 1], False])

    avg_input = leaf(ramp(32), (1, 2, 4, 4))
    case("aten::avg_pool2d.default", [avg_input, [2, 2], [2, 2], [0, 0], False, True])
    avg_input = leaf(ramp(18), (1, 2, 3, 3))
    case("aten::avg_pool2d.default", [avg_input, [2, 2], [1, 1], [1, 1], False, True])
    avg_input = leaf(ramp(18), (1, 2, 3, 3))
    case("aten::avg_pool2d.default", [avg_input, [2, 2], [1, 1], [1, 1], False, False])

    pad_input = leaf(ramp(18), (1, 2, 3, 3))
    case("aten::constant_pad_nd.default", [pad_input, [1, 1, 0, 0], 0.5])
    pad_input = leaf(ramp(6), (2, 3))
    case("aten::constant_pad_nd.default", [pad_input, [0, 2, 1, 0], -1.0])

    flip_input = leaf(ramp(6), (2, 3))
    case("aten::flip.default", [flip_input, [0, 1]])
    flip_input = leaf(ramp(6), (2, 3))
    case("aten::flip.default", [flip_input, [1]])

    gather_self = leaf(ramp(9), (3, 3))
    gather_index = tensor([0, 2, 1, 2, 0, 1], dtype="int64").reshape(2, 3)
    case("aten::gather.default", [gather_self, 1, gather_index])
    gather_self = leaf(ramp(9), (3, 3))
    gather_index = tensor([2, 0, 1], dtype="int64").reshape(3, 1)
    case("aten::gather.default", [gather_self, 0, gather_index])

    select_self = leaf(ramp(9), (3, 3))
    select_index = tensor([2, 0], dtype="int64")
    case("aten::index_select.default", [select_self, 0, select_index])
    select_self = leaf(ramp(9), (3, 3))
    case("aten::index_select.default", [select_self, 1, select_index])

    batch_input = leaf(ramp(18), (1, 2, 3, 3))
    batch_weight = leaf([1.5, 0.5])
    batch_bias = leaf([0.1, -0.1])
    batch_mean = tensor([0.1, -0.2], dtype="float64")
    batch_var = tensor([1.0, 0.5], dtype="float64")
    case("aten::batch_norm.default",
         [batch_input, batch_weight, batch_bias, batch_mean, batch_var, False, 0.1, 1e-5])
    train_input = leaf(ramp(8), (1, 2, 2, 2))
    train_weight = leaf([1.5, 0.5])
    train_bias = leaf([0.1, -0.1])
    train_mean = tensor([0.1, -0.2], dtype="float64")
    train_var = tensor([1.0, 0.5], dtype="float64")
    case("aten::batch_norm.default",
         [train_input, train_weight, train_bias, train_mean, train_var, True, 0.1, 1e-5])

    slice_input = leaf(ramp(8), (2, 4))
    case("aten::slice.Tensor", [slice_input, 1, 1, 4, 2])
    slice_input = leaf(ramp(8), (2, 4))
    case("aten::slice.Tensor", [slice_input, 1, None, None, 1])
    slice_input = leaf(ramp(8), (2, 4))
    case("aten::slice.Tensor", [slice_input, 0, -1, None, 1])

    scatter_self = leaf(ramp(6), (2, 3))
    scatter_index = tensor([0, 2, 1, 0], dtype="int64").reshape(2, 2)
    scatter_src = leaf([10.0, 11.0, 12.0, 13.0], (2, 2))
    case("aten::scatter.default", [scatter_self, 1, scatter_index, scatter_src])

    layer_input = leaf(ramp(24), (2, 3, 4))
    layer_weight = leaf([1.0, 1.5, 0.5, 2.0])
    layer_bias = leaf([0.1, -0.1, 0.2, -0.2])
    case("aten::layer_norm.default", [layer_input, [4], layer_weight, layer_bias, 1e-5])

    group_input = leaf(ramp(16), (1, 4, 2, 2))
    group_weight = leaf([1.0, 0.5, 1.5, 2.0])
    group_bias = leaf([0.1, -0.1, 0.2, -0.2])
    case("aten::group_norm.default", [group_input, 2, group_weight, group_bias, 1e-5])

    # ------------------------------------------------------------------ combination
    case("aten::cat.default", [[A(), Positive6()], 1])
    case("aten::stack.default", [[A(), Positive6()], 0])

    # ------------------------------------------------------------------ losses
    target = tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], dtype="float64").reshape(2, 3)
    case("aten::mse_loss.default", [A(), target, "mean"])
    logits = leaf([1.0, 2.0, 0.5, -1.0, 0.0, 1.5, 2.5, 0.5, -0.5, 1.0, 0.0, 2.0], (3, 4))
    labels = tensor([0, 2, 1], dtype="int64")
    case("aten::cross_entropy.default", [logits, labels, "mean"])
    logits = leaf([1.0, 2.0, 0.5, -1.0, 0.0, 1.5, 2.5, 0.5, -0.5, 1.0, 0.0, 2.0], (3, 4))
    case("aten::cross_entropy.default", [logits, labels, "sum"])
    logits = leaf([1.0, 2.0, 0.5, -1.0, 0.0, 1.5, 2.5, 0.5, -0.5, 1.0, 0.0, 2.0], (3, 4))
    case("aten::cross_entropy.default", [logits, labels, "none"])

    # ------------------------------------------------------------------ in-place
    def inplace_args(operator):
        self_inplace = make_tensor_arg([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(2, 3)
        other = make_tensor_arg([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]).reshape(2, 3)
        if operator in ("aten::add_.Tensor", "aten::sub_.Tensor"):
            return [self_inplace, other, 1]
        if operator in ("aten::mul_.Tensor", "aten::div_.Tensor", "aten::copy_.default"):
            return [self_inplace, other]
        if operator == "aten::fill_.Scalar":
            return [self_inplace, 2.5]
        return [self_inplace]

    for operator in (
        "aten::add_.Tensor",
        "aten::sub_.Tensor",
        "aten::mul_.Tensor",
        "aten::div_.Tensor",
        "aten::copy_.default",
        "aten::zero_.default",
        "aten::fill_.Scalar",
    ):
        case(operator, inplace_args(operator), inplace=True, grads=False)


def call_case(spec):
    operator = spec["operator"]
    args = list(spec["args"])
    name = operator.split("::", 1)[1]
    base, _, overload = name.partition(".")
    overload = overload or "default"

    if operator == "aten::zeros.default":
        return [torch.zeros(int(args[0][0]), int(args[0][1]), dtype=args[1], device=args[2])]
    if operator == "aten::ones.default":
        return [torch.ones(int(args[0][0]), int(args[0][1]), dtype=args[1], device=args[2])]
    if operator == "aten::empty.default":
        return [torch.empty(int(args[0][0]), int(args[0][1]), dtype=args[1], device=args[2])]
    if operator == "aten::full.default":
        return [torch.full(tuple(args[0]), args[1], dtype=args[2], device=args[3])]
    if operator == "aten::arange.default":
        return [torch.arange(args[0], args[1], args[2], dtype=args[3], device=args[4])]
    if operator == "aten::randn.default":
        return [torch.randn(tuple(args[0]), dtype=args[1], device=args[2])]
    if operator == "aten::rand.default":
        return [torch.rand(tuple(args[0]), dtype=args[1], device=args[2])]
    if operator == "aten::randint.default":
        return [torch.randint(args[0], args[1], tuple(args[2]), dtype=args[3], device=args[4])]
    if operator == "aten::eye.default":
        return [torch.eye(args[0], dtype=args[1], device=args[2])]
    if operator == "aten::reshape.default":
        return [args[0].reshape(tuple(args[1]))]
    if operator == "aten::view.default":
        return [args[0].view(tuple(args[1]))]
    if operator == "aten::transpose.default":
        return [args[0].transpose(args[1], args[2])]
    if operator == "aten::permute.default":
        return [args[0].permute(tuple(args[1]))]
    if operator == "aten::narrow.default":
        return [args[0].narrow(args[1], args[2], args[3])]
    if operator == "aten::select.default":
        return [args[0].select(args[1], args[2])]
    if operator == "aten::squeeze.default":
        return [args[0].squeeze()]
    if operator == "aten::squeeze.dim":
        return [args[0].squeeze(args[1])]
    if operator == "aten::unsqueeze.default":
        return [args[0].unsqueeze(args[1])]
    if operator == "aten::expand.default":
        return [args[0].expand(tuple(args[1]))]
    if operator == "aten::flatten.default":
        return [args[0].flatten(args[1], args[2])]
    if operator == "aten::t.default":
        return [args[0].t()]
    if operator == "aten::contiguous.default":
        return [args[0].contiguous()]
    if operator == "aten::clone.default":
        return [args[0].clone()]
    if operator == "aten::detach.default":
        return [args[0].detach()]
    if operator == "aten::to.dtype":
        return [args[0].to(args[1])]
    if operator in (
        "aten::add.Tensor", "aten::sub.Tensor", "aten::mul.Tensor", "aten::div.Tensor",
        "aten::pow.Tensor_Tensor", "aten::maximum.default", "aten::minimum.default",
        "aten::eq.Tensor", "aten::ne.Tensor", "aten::lt.Tensor", "aten::le.Tensor",
        "aten::gt.Tensor", "aten::ge.Tensor",
    ):
        lhs, rhs = args[0], args[1]
        op = {
            "aten::add.Tensor": torch.add,
            "aten::sub.Tensor": torch.sub,
            "aten::mul.Tensor": torch.mul,
            "aten::div.Tensor": torch.div,
            "aten::pow.Tensor_Tensor": torch.pow,
            "aten::maximum.default": torch.maximum,
            "aten::minimum.default": torch.minimum,
            "aten::eq.Tensor": torch.eq,
            "aten::ne.Tensor": torch.ne,
            "aten::lt.Tensor": torch.lt,
            "aten::le.Tensor": torch.le,
            "aten::gt.Tensor": torch.gt,
            "aten::ge.Tensor": torch.ge,
        }[operator]
        if operator in ("aten::add.Tensor", "aten::sub.Tensor"):
            return [op(lhs, rhs, alpha=args[2])]
        return [op(lhs, rhs)]
    if operator == "aten::pow.Tensor_Scalar":
        return [torch.pow(args[0], args[1])]
    if operator == "aten::logical_not.default":
        return [torch.logical_not(args[0])]
    unary = {
        "aten::neg.default": torch.neg,
        "aten::abs.default": torch.abs,
        "aten::square.default": torch.square,
        "aten::sqrt.default": torch.sqrt,
        "aten::exp.default": torch.exp,
        "aten::log.default": torch.log,
        "aten::tanh.default": torch.tanh,
        "aten::sigmoid.default": torch.sigmoid,
        "aten::relu.default": torch.relu,
        "aten::reciprocal.default": torch.reciprocal,
        "aten::floor.default": torch.floor,
        "aten::ceil.default": torch.ceil,
        "aten::sign.default": torch.sign,
        "aten::sin.default": torch.sin,
        "aten::cos.default": torch.cos,
        "aten::log1p.default": torch.log1p,
    }
    if operator in unary:
        return [unary[operator](args[0])]
    if operator == "aten::clamp.default":
        return [torch.clamp(args[0], min=args[1], max=args[2])]
    if operator == "aten::sum.default":
        return [args[0].sum()]
    if operator == "aten::sum.dim_IntList":
        return [args[0].sum(dim=tuple(args[1]), keepdim=args[2])]
    if operator == "aten::mean.default":
        return [args[0].mean()]
    if operator == "aten::mean.dim":
        return [args[0].mean(dim=tuple(args[1]), keepdim=args[2])]
    if operator == "aten::prod.default":
        return [args[0].prod()]
    if operator == "aten::prod.dim":
        return [args[0].prod(dim=args[1], keepdim=args[2])]
    if operator == "aten::max.default":
        return [args[0].max()]
    if operator == "aten::min.default":
        return [args[0].min()]
    if operator == "aten::max.dim":
        return list(args[0].max(dim=args[1], keepdim=args[2]))
    if operator == "aten::min.dim":
        return list(args[0].min(dim=args[1], keepdim=args[2]))
    if operator == "aten::argmax.default":
        return [args[0].argmax() if args[1] is None else args[0].argmax(dim=args[1], keepdim=args[2])]
    if operator == "aten::argmin.default":
        return [args[0].argmin() if args[1] is None else args[0].argmin(dim=args[1], keepdim=args[2])]
    if operator == "aten::softmax.default":
        return [torch.softmax(args[0], dim=args[1])]
    if operator == "aten::log_softmax.default":
        return [torch.log_softmax(args[0], dim=args[1])]
    if operator == "aten::matmul.default":
        return [args[0].matmul(args[1])]
    if operator == "aten::mm.default":
        return [args[0].mm(args[1])]
    if operator == "aten::bmm.default":
        return [args[0].bmm(args[1])]
    if operator == "aten::addmm.default":
        return [torch.addmm(args[0], args[1], args[2], beta=args[3], alpha=args[4])]
    if operator == "aten::linear.default":
        return [F.linear(args[0], args[1], args[2])]
    if operator == "aten::conv2d.default":
        return [F.conv2d(args[0], args[1], args[2], stride=tuple(args[3]), padding=tuple(args[4]),
                         dilation=tuple(args[5]), groups=args[6])]
    if operator == "aten::max_pool2d.default":
        return [F.max_pool2d(args[0], kernel_size=tuple(args[1]), stride=tuple(args[2]),
                             padding=tuple(args[3]), dilation=tuple(args[4]), ceil_mode=args[5])]
    if operator == "aten::avg_pool2d.default":
        return [F.avg_pool2d(args[0], kernel_size=tuple(args[1]), stride=tuple(args[2]),
                             padding=tuple(args[3]), ceil_mode=args[4], count_include_pad=args[5])]
    if operator == "aten::constant_pad_nd.default":
        return [F.pad(args[0], list(args[1]), mode="constant", value=args[2])]
    if operator == "aten::flip.default":
        return [torch.flip(args[0], tuple(args[1]))]
    if operator == "aten::gather.default":
        return [torch.gather(args[0], args[1], args[2])]
    if operator == "aten::index_select.default":
        return [torch.index_select(args[0], args[1], args[2])]
    if operator == "aten::slice.Tensor":
        selection = [slice(None)] * len(args[0].shape)
        selection[args[1]] = slice(args[2], args[3], args[4])
        return [args[0][tuple(selection)]]
    if operator == "aten::scatter.default":
        return [torch.scatter(args[0], args[1], args[2], args[3])]
    if operator == "aten::layer_norm.default":
        return [F.layer_norm(args[0], tuple(args[1]), weight=args[2], bias=args[3], eps=args[4])]
    if operator == "aten::group_norm.default":
        return [F.group_norm(args[0], args[1], weight=args[2], bias=args[3], eps=args[4])]
    if operator == "aten::batch_norm.default":
        return [F.batch_norm(args[0], args[3], args[4], weight=args[1], bias=args[2],
                             training=args[5], momentum=args[6], eps=args[7])]
    if operator == "aten::cat.default":
        return [torch.cat(tuple(args[0]), dim=args[1])]
    if operator == "aten::stack.default":
        return [torch.stack(tuple(args[0]), dim=args[1])]
    if operator == "aten::mse_loss.default":
        return [F.mse_loss(args[0], args[1], reduction=args[2])]
    if operator == "aten::cross_entropy.default":
        return [F.cross_entropy(args[0], args[1], reduction=args[2])]
    if operator == "aten::add_.Tensor":
        args[0].add_(args[1], alpha=args[2])
        return [args[0]]
    if operator == "aten::sub_.Tensor":
        args[0].sub_(args[1], alpha=args[2])
        return [args[0]]
    if operator == "aten::mul_.Tensor":
        args[0].mul_(args[1])
        return [args[0]]
    if operator == "aten::div_.Tensor":
        args[0].div_(args[1])
        return [args[0]]
    if operator == "aten::copy_.default":
        args[0].copy_(args[1])
        return [args[0]]
    if operator == "aten::zero_.default":
        args[0].zero_()
        return [args[0]]
    if operator == "aten::fill_.Scalar":
        args[0].fill_(args[1])
        return [args[0]]
    raise NotImplementedError(operator)


def slug(operator, index):
    name = operator.split("::", 1)[1]
    return f"{name}-{index:03d}"


def main():
    if not OUT.exists():
        OUT.mkdir(parents=True, exist_ok=True)
    for existing in OUT.glob("*.json"):
        existing.unlink()

    torch.manual_seed(20261031)
    build_cases()
    torch_version = torch.__version__
    generated = []
    for index, spec in enumerate(CASES):
        encoded_args = [encode_value(value) for value in spec["args"]]
        outputs = call_case(spec)
        outputs = [o for o in outputs if isinstance(o, torch.Tensor)]
        gradient_entries = []
        if spec["grads"] and not spec["nondeterministic"] and not spec["inplace"]:
            floating = [o for o in outputs if o.is_floating_point() and o.requires_grad]
            if floating:
                loss = sum(o.sum() for o in floating)
                if loss.requires_grad:
                    loss.backward()
                    for arg_index, arg in enumerate(spec["args"]):
                        if isinstance(arg, torch.Tensor) and arg.grad is not None:
                            gradient_entries.append({"argIndex": arg_index, "grad": encode_tensor(arg.grad)})
        fixture = {
            "operator": spec["operator"],
            "torch": torch_version,
            "python": sys.version.split()[0],
            "nondeterministic": spec["nondeterministic"],
            "inplace": spec["inplace"],
            "note": spec["note"],
            "args": encoded_args,
            "outputs": [encode_tensor(o, include_data=not spec["nondeterministic"]) for o in outputs],
            "grads": gradient_entries,
        }
        filename = slug(spec["operator"], index) + ".json"
        with open(OUT / filename, "w", encoding="utf-8") as handle:
            json.dump(fixture, handle, indent=2)
            handle.write("\n")
        generated.append(
            {
                "file": filename,
                "operator": spec["operator"],
                "nondeterministic": spec["nondeterministic"],
                "inplace": spec["inplace"],
            }
        )

    with open(OUT / "index.json", "w", encoding="utf-8") as handle:
        json.dump({"torch": torch_version, "cases": generated}, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {len(generated)} fixtures with torch {torch_version} to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
