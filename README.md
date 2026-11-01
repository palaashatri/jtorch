# JTorch

PyTorch-style tensors, autograd and neural networks in pure Java (JDK 25): exact PyTorch semantics for shapes, strides, views and gradients, a CPU reference backend with no native dependencies, and planned optional native/GPU backends through the Foreign Function & Memory API instead of JNI.

**Status: M0–M2 are complete and most of M3 is implemented and tested on the host.** This is not a PyTorch-class 1.0 and does not claim to be: CUDA, distributed training, the compiler, advanced indexing/convolution, CNN/real-MNIST demos and the remaining applications (M4–M16) are not implemented. M3 still lacks a CNN, samplers/prefetch, real-MNIST acceptance and the UI/accessibility gate. See [compat/](compat/) for the machine-readable ledger and [ADR 0005](docs/adr/0005-supported-scope-m0-m3.md) for the precise scope.

- Design, architecture, testing strategy and milestones: [docs/DESIGN.md](docs/DESIGN.md)
- Contributor and coding-agent instructions: [AGENTS.md](AGENTS.md)
- Generated operator table: [docs/OPERATORS.md](docs/OPERATORS.md)

## What works today

- **Tensors:** dense strided CPU and meta tensors for `float32`, `float64`, `int32`, `int64` and `bool`; immutable `Shape`/`Strides`; FFM `MemorySegment` storage with Cleaner-based release; mapped-file tensors; exact `int64` arithmetic (use `itemLong()` for integer scalars; `item()` refuses values a double cannot represent); views (`reshape`, `view`, `transpose`, `permute`, `narrow`, `select`, `squeeze`, `unsqueeze`, `expand`, `flatten`, `t`, `contiguous`, `clone`, `detach`) with shared storage and version counters.
- **Dispatcher:** priority-ordered `DispatchKeySet` with `AUTOGRAD`, `META`, `CPU` and `COMPOSITE` keys; schema-validated typed arguments; per-key registration; composite decompositions; duplicate registration and missing-coverage failures.
- **Autograd:** reverse-mode engine with topological traversal, gradient accumulation, broadcast reduction, saved-tensor version validation, tensor hooks, `retainGraph`, and explicit rejection of `create_graph`.
- **Operators:** 87 schema-declared operators (`aten` namespace) with forward, meta and backward coverage where declared. Every operator has a PyTorch 2.11.0 fixture; 93 fixtures contain exact values and gradients, while `empty`/`rand`/`randn`/`randint` fixtures are metadata-only because JTorch does not promise PyTorch's random stream. The parity runner checks values, shapes, strides, storage offsets, dtypes, gradients and meta/CPU agreement.
- **Neural networks:** `Module` with explicit parameter/buffer/child registration, state dictionaries, strict/non-strict loading, forward hooks and train/eval; `Linear`, `ReLU`, `Sigmoid`, `Tanh`, `Flatten`, `Sequential`; MSE and cross-entropy criteria.
- **Optimization:** SGD (momentum, weight decay, dampening, Nesterov), Adam, AdamW, StepLR and ExponentialLR; optimizer updates match PyTorch fixtures.
- **Data:** tensor datasets, deterministic shuffling and batching, an MNIST IDX reader, and a deterministic ten-class synthetic digit dataset for demos/tests.
- **Serialization:** checksummed native checkpoints (JSON manifest + little-endian payload + SHA-256) and Safetensors read/write, both validated for corruption, truncation and offset tampering. No pickle, ever.
- **TensorLab:** a Swing training workbench and a headless smoke test that trains a small MLP, saves a checkpoint, loads it into a fresh training object and requires the resumed epoch to be exactly equal to the uninterrupted one. `demoPackage` builds a jpackage app image and runs the packaged launcher.

Java array overloads choose the dtype from the primitive type: `float[]` is float32, `double[]` is float64, `long[]` is int64, `int[]` is int32 and `boolean[]` is bool. This differs from Python literal defaults, which are float32; pass `TensorOptions` when a specific dtype is required.

Not implemented yet: CUDA/ROCm/Metal/Vulkan, distributed training, the compiler/export pipeline, AMP/quantization, float16/bfloat16/complex dtypes, sparse/nested layouts, convolution/pooling/normalization, advanced indexing, vmap/JVP/VJP/Jacobian/Hessian/functionalization, in-place autograd on tensors that require grad, higher-order gradients, optimizer state checkpointing, and real-MNIST accuracy claims. Each is recorded as unsupported rather than silently emulated.

## Commands

JDK 25 and the checked-in Gradle 9.8.2 wrapper are required. There are no third-party runtime dependencies; JUnit is test-only.

```bash
./gradlew clean check demoCheck      # gradlew.bat on Windows
./gradlew parityFast                 # differential PyTorch fixtures, all operators
./gradlew check demoCheck demoPackage
./gradlew regenerateSchemas verifyGeneratedClean
./gradlew :examples:vertical-slice:run
```

The PyTorch oracle is optional; the generated fixtures are checked in and the build never needs Python:

```bash
python3 tools/oracle/generate.py
python3 tools/oracle/optimizers.py
python3 tools/oracle/safetensors_fixture.py
```

`compat/baseline.toml` pins PyTorch 2.11.0 and Python 3.13.9.

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
- `modules/jtorch-nn`, `jtorch-optim`, `jtorch-data`, `jtorch-serialization` — training stack
- `schema/`, `tools/opgen` — canonical operator schema and deterministic generator
- `tools/oracle` — pinned PyTorch fixture generators (not used at runtime)
- `tests/parity` — golden fixtures and the generic differential runner
- `demos/tensorlab` — training workbench and packaged smoke test
- `compat/` — generated operator ledger, scope ledgers and the oracle baseline
- `docs/adr/` — architecture decision records

## How this differs from existing JVM options

Deep Java Library (DJL) and the Bytedeco PyTorch presets call the native PyTorch/LibTorch library; Deeplearning4j has its own native engine and API. JTorch implements PyTorch semantics on the JVM itself, with a pure-Java reference backend and FFM (not JNI) for future native acceleration.

## Licence

No licence has been chosen yet, so the code is not yet open source in the legal sense. Add a `LICENSE` file (the design expects one, plus `NOTICE`) before accepting outside contributions.

## Independent JTorch library and edge efficiency

**JTorch is not a candidate for bundled SLOPOS/PARAMA desktop applications or their curated first-party Software catalogs.** It remains an independently distributed Java numerical library with its own examples and planned demos; no new desktop icon, OS-specific app wrapper or PARAMA port is required. If a Linux/SLOPOS device supports Java 25, library usage follows ordinary Java deployment.

The Raspberry Pi Zero 2 W remains an **aspirational small-tensor efficiency reference**, conditional on a compatible JDK. Prioritize bounded allocation, correct views/gradients, CPU-first operation, low memory footprint and honest rejection of out-of-memory tensor shapes. Keep measured resource tests distinct from large-model promises and from unverified board/OS compatibility. See [AGENTS.md](AGENTS.md#independent-library-portability-and-raspberry-pi-zero-2-w-efficiency).
