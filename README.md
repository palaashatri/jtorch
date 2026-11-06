# JTorch

PyTorch-style tensors, autograd and neural networks in pure Java (JDK 25): exact PyTorch semantics for shapes, strides, views and gradients, a CPU reference backend with no native dependencies, and planned optional native/GPU backends through the Foreign Function & Memory API instead of JNI.

**Status: M0–M4 are complete for the automated reference scope; manual screen-reader/GUI acceptance has not been performed.** This is not a PyTorch-class 1.0 and does not claim to be: CUDA, distributed training, the compiler and the remaining applications (M5–M16) are not implemented; 2D/nD FFT and ONNX export/external data are explicitly out of scope. The CNN path, prefetch, accessibility metadata/keyboard shortcuts and real-MNIST acceptance test are implemented; the synthetic TensorLab check plus the real-IDX CNN check are both runnable. See [compat/](compat/) for the machine-readable ledger and [ADRs 0005–0006](docs/adr/) for the precise scope.

- Design, architecture, testing strategy and milestones: [docs/DESIGN.md](docs/DESIGN.md)
- Contributor and coding-agent instructions: [AGENTS.md](AGENTS.md)
- Generated operator table: [docs/OPERATORS.md](docs/OPERATORS.md)

## What works today

- **Tensors:** dense strided CPU and meta tensors for `float32`, `float64`, `int32`, `int64`, `bool` and `complex64`/`complex128` (interleaved real/imag, FFT only); immutable `Shape`/`Strides`; FFM `MemorySegment` storage with Cleaner-based release; mapped-file tensors; exact `int64` arithmetic (use `itemLong()` for integer scalars; `item()` refuses values a double cannot represent); views (`reshape`, `view`, `transpose`, `permute`, `narrow`, `select`, `squeeze`, `unsqueeze`, `expand`, `flatten`, `t`, `contiguous`, `clone`, `detach`) with shared storage and version counters.
- **Dispatcher:** priority-ordered `DispatchKeySet` with `AUTOGRAD`, `META`, `CPU` and `COMPOSITE` keys; schema-validated typed arguments; per-key registration; composite decompositions; duplicate registration and missing-coverage failures.
- **Autograd:** reverse-mode engine with topological traversal, gradient accumulation, broadcast reduction, saved-tensor version validation, tensor hooks, `retainGraph`, and explicit rejection of `create_graph`.
- **Operators:** 107 schema-declared public operators (`aten` namespace) with forward, meta and backward coverage where declared (86 `full-semantics`, 16 `autograd`, 5 `forward`). Every operator has a PyTorch 2.11.0 fixture; 143 fixtures contain exact values and gradients, while `empty`/`rand`/`randn`/`randint` fixtures are metadata-only because JTorch does not promise PyTorch's random stream. The parity runner checks values, shapes, strides, storage offsets, dtypes, gradients and meta/CPU agreement.
- **Neural networks:** `Module` with explicit parameter/buffer/child registration, state dictionaries, strict/non-strict loading, forward hooks and train/eval; `Linear`, `Conv2d` (including groups), `MaxPool2d`/`AvgPool2d` (ceil mode and divisor override), `AdaptiveAvgPool2d`, `BatchNorm2d`, `LayerNorm`, `GroupNorm`, `Embedding`, `GELU`, `MultiheadAttention`, `TransformerEncoderLayer`, `ReLU`, `Sigmoid`, `Tanh`, `Flatten`, `Sequential`; MSE and cross-entropy criteria.
- **Optimization:** SGD (momentum, weight decay, dampening, Nesterov), Adam, AdamW, StepLR and ExponentialLR; optimizer updates match PyTorch fixtures.
- **Data:** tensor datasets, sequential/random/batch samplers, deterministic shuffling and batching, bounded virtual-thread prefetch, an MNIST IDX reader (including gzip), and a deterministic ten-class synthetic digit dataset for demos/tests.
- **Serialization:** checksummed native checkpoints (JSON manifest + little-endian payload + SHA-256) and Safetensors read/write, both validated for corruption, truncation and offset tampering. No pickle, ever. ONNX import foundations (`interop/jtorch-onnx`, opset 13–23, import-only) execute the documented subset with unsupported operators reported before execution; export and external data are not implemented.
- **M4:** convolution with groups; max/average pooling with ceil mode and `divisor_override`; adaptive pooling; `interpolate` (nearest/bilinear); `gelu` (exact and tanh); `embedding` (padding and frequency scaling); batch/layer/group normalization; gathering, slicing, scattering, padding and flipping with reverse-mode derivatives where defined; 1D FFT (`fft`/`ifft`/`rfft`/`irfft` with `n`/`dim`/`norm`, permuted output strides, reverse-mode adjoints); multi-head attention and post-norm transformer encoder layers; ResNet9/ViT fixture models (Safetensors); `vision-inspector` demo (single-image and batch classification, gradient saliency, JSON/CSV export, corrupt-image isolation, off-EDT workers with decoded-size limits, packaged headless smoke test). `scatter` forward accepts a larger `src` like PyTorch but its backward rejects that non-parity case explicitly.
- **TensorLab:** a Swing training workbench with accessibility names, descriptions, mnemonics and keyboard shortcuts, and headless smoke tests for an MLP and a CNN. The synthetic MLP/CNN checks train, save a checkpoint, load it into a fresh training object and require the resumed epoch to be exactly equal to the uninterrupted one. `--mnist-dir` runs the same CNN over real IDX files; the acceptance threshold is 0.95 on held-out samples, and the host run reached 0.956 on 1,000 test images after four epochs over 12,000 training images. `demoPackage` builds a jpackage app image and runs the packaged launcher.

Java array overloads choose the dtype from the primitive type: `float[]` is float32, `double[]` is float64, `long[]` is int64, `int[]` is int32 and `boolean[]` is bool. This differs from Python literal defaults, which are float32; pass `TensorOptions` when a specific dtype is required.

Not implemented yet: CUDA/ROCm/Metal/Vulkan, distributed training, the compiler/export pipeline, AMP/quantization, float16/bfloat16 dtypes, sparse/nested layouts, 2D/nD FFT, ONNX export/external data, vmap/JVP/VJP/Jacobian/Hessian/functionalization, in-place autograd on tensors that require grad, higher-order gradients, optimizer state checkpointing, and manual screen-reader acceptance. Convolution/pooling are NCHW (pooling also accepts CHW); complex arithmetic outside FFT is rejected explicitly; unsupported configurations fail explicitly. Everything else is recorded as unsupported rather than silently emulated.

## Commands

JDK 25 and the checked-in Gradle 9.8.2 wrapper are required. There are no third-party runtime dependencies; JUnit is test-only.

```bash
./gradlew clean check demoCheck      # gradlew.bat on Windows
./gradlew parityFast                 # differential PyTorch fixtures, all operators
./gradlew check demoCheck demoPackage
./gradlew regenerateSchemas verifyGeneratedClean
./gradlew :examples:vertical-slice:run
./gradlew :demos:tensorlab:mnistCheck -PmnistDir=/path/to/mnist-idx
./gradlew :demos:vision-inspector:demoCheck
./gradlew :demos:vision-inspector:demoPackage
./gradlew :interop:jtorch-onnx:test
```

The PyTorch oracle is optional; the generated fixtures are checked in and the build never needs Python:

```bash
python3 tools/oracle/generate.py
python3 tools/oracle/optimizers.py
python3 tools/oracle/safetensors_fixture.py
python3 tools/oracle/vision_fixture.py
```

`compat/baseline.toml` pins PyTorch 2.11.0 and Python 3.13.17.

## The first vertical slice

```java
Tensor x = Torch.tensor(new float[] {1, 2, 3, 4}, Shape.of(2, 2)).requiresGrad(true);
Tensor w = Torch.randn(Shape.of(2, 1), TensorOptions.defaults().requiresGrad(true));
Tensor loss = x.matmul(w).square().mean();
loss.backward();
SGD.builder().parameters(List.of(new Parameter(w))).learningRate(1e-2).build().step();
```

This is executable in [`examples/vertical-slice`](examples/vertical-slice) and its values and gradients are checked against PyTorch fixtures.

## Repository layout

- `modules/jtorch-core` — value types, storage, tensors, dispatcher, schema handles
- `modules/jtorch-autograd` — reverse-mode engine and native backward formulas
- `backends/jtorch-cpu-reference` — CPU, meta and composite kernels
- `modules/jtorch-api` — public `Torch` facade
- `modules/jtorch-nn`, `jtorch-optim`, `jtorch-data`, `jtorch-serialization`, `jtorch-vision` — training and vision stack
- `interop/jtorch-onnx` — ONNX import foundations (opset 13–23, no third-party runtime deps)
- `schema/`, `tools/opgen` — canonical operator schema and deterministic generator
- `tools/oracle` — pinned PyTorch fixture generators (not used at runtime)
- `tests/parity` — golden fixtures and the generic differential runner
- `demos/tensorlab` — training workbench and packaged smoke test
- `demos/vision-inspector` — image classification, batch runs and saliency
- `compat/` — generated operator ledger, scope ledgers and the oracle baseline
- `docs/adr/` — architecture decision records

## How this differs from existing JVM options

Deep Java Library (DJL) and the Bytedeco PyTorch presets call the native PyTorch/LibTorch library; Deeplearning4j has its own native engine and API. JTorch implements PyTorch semantics on the JVM itself, with a pure-Java reference backend and FFM (not JNI) for future native acceleration.

## Licence

No licence has been chosen yet, so the code is not yet open source in the legal sense. Add a `LICENSE` file (the design expects one, plus `NOTICE`) before accepting outside contributions.

