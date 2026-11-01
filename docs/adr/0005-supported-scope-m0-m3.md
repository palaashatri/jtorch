# ADR 0005: Declared scope and honest limitations for M0–M3

- Status: accepted
- Date: 2026-11-01

## Context

The roadmap runs to M16 (CUDA, distributed training, compiler, diffusion, Python facade). The reference build in this
repository implements the first vertical slice and the M1–M3 training stack. Calling that "100% complete" would
violate the project's no-false-completeness rule.

## Decision

The repository reports exactly what is implemented and tested, and records everything else as `unsupported` in
`compat/`. M0–M2 are complete; M3 is partially complete (no CNN, real-MNIST acceptance, sampler/prefetch breadth or
UI/accessibility gate):

Implemented and oracle-tested (PyTorch 2.11.0):

- dense strided CPU and meta tensors for float32/float64/int32/int64/bool;
- 87 operators with forward, shape/stride metadata, meta execution and (where declared) reverse-mode gradients;
- views, storage aliasing, shared version counters and saved-tensor mutation detection;
- `Linear`/`Sequential`/activation modules, MSE/cross-entropy, SGD/Adam/AdamW, StepLR/ExponentialLR;
- tensor datasets, deterministic batching, MNIST IDX reader and a synthetic ten-class demo dataset;
- native checkpoints with SHA-256 validation and Safetensors read/write;
- TensorLab headless training with checkpoint resume and a packaged jpackage smoke test (macOS host proven).

Explicitly unsupported in this build:

- CUDA/ROCm/Metal/Vulkan execution, distributed training, the compiler/export pipeline, AMP/quantization;
- float16, bfloat16, complex, float8 and quantized dtypes; sparse/nested layouts;
- in-place autograd on tensors that require grad (a clear error is raised; `noGrad` is required);
- `create_graph`/higher-order gradients, forward AD, vmap/JVP/VJP/Jacobian/Hessian/functionalization;
- advanced indexing, gather/scatter, convolution, pooling and normalization;
- reverse-mode gradients for `prod` (forward-only), and `cross_entropy` beyond 2-D logits;
- optimizer state serialization and parameter groups;
- real MNIST accuracy claims: the demo uses a labelled synthetic dataset unless IDX files are supplied.

## Consequences

- README and `compat/` must not call the framework "PyTorch-class" or "drop-in".
- Each unsupported item has a clear failure mode rather than a silent CPU fallback or dtype coercion.
- Advancing to M4 requires new oracle fixtures, new ledger entries and new module work; the milestone backlog in
  `docs/DESIGN.md` remains the plan.
