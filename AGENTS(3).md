# AGENTS.md — JTorch JDK 25 Project Constitution, Architecture, Demo Suite, and One-Shot Execution Plan

> **Purpose:** This single file is the permanent project constitution, end-state technical design, demo-application specification, CI/release design, and one-shot execution contract for Codex and other coding agents working on JTorch.
>
> **Instruction precedence:** User instructions override this file. A more-specific nested `AGENTS.md` may add local rules but must not weaken repository-wide correctness, compatibility, security, or test gates.

---

## 0. One-shot directive for Codex and all coding agents

You are the lead architect, implementer, integrator, and release engineer for JTorch. This file is the repository-wide source of truth. Do not stop after writing plans, generating empty modules, or producing architecture prose. Inspect what exists, preserve correct work, and then implement the earliest incomplete vertical slice to its exit criteria.

### 0.1 First-run procedure

On the first run:

1. Read this file completely before editing source.
2. Inspect the repository, Git history, issues, build scripts, generated code, tests, native sources, workflows, and existing demo applications.
3. Establish the current milestone by evidence. Do not infer completion from filenames or stubs.
4. Ensure the checked-in Gradle wrapper is at least Gradle 9.1 and is verified to run on JDK 25. Prefer the latest repository-approved Gradle 9.x release, recorded in the wrapper properties.
5. Ensure `./gradlew check` succeeds on JDK 25 before broad feature work. On Windows, the equivalent command is `gradlew.bat check`.
6. Create a concise machine-readable implementation status under `compat/` and GitHub issues. Do not create duplicate design documents that drift from this file.
7. Decompose work into bounded tasks and delegate independent tasks to subagents. Use isolated Git worktrees or branches for overlapping implementation work.
8. Keep architecture, public API integration, generated-schema changes, and final merge decisions in the lead thread.
9. Implement complete vertical slices: schema, public API, dispatcher registration, meta behavior, backend kernel, autograd formula, tests, documentation, benchmark, and at least one consuming demo path where applicable.
10. Build and test JTorch and every currently declared demo application on the host platform before concluding.
11. Run the relevant GitHub-workflow-equivalent Gradle tasks locally and report exact commands and outcomes.
12. Leave the repository buildable. Record remaining tasks as small, dependency-ordered issues rather than vague TODO prose.

Do not ask the user to resolve choices already specified here. When a genuinely new irreversible decision appears, create an ADR under `docs/adr/`, choose the safest reversible default, implement the narrowest correct path, and continue.

### 0.2 Required first vertical slice

Unless already implemented and verified, the first integrated slice is a CPU training path on JDK 25:

```java
Tensor x = Torch.tensor(new float[] {1, 2, 3, 4}, Shape.of(2, 2))
        .requiresGrad(true);
Tensor w = Torch.randn(Shape.of(2, 1), TensorOptions.defaults()
        .dtype(DType.FLOAT32)
        .device(Device.cpu())
        .requiresGrad(true));

Tensor loss = x.matmul(w).square().mean();
loss.backward();

Optimizer optimizer = SGD.builder()
        .parameters(List.of(new Parameter(w)))
        .learningRate(1e-2)
        .build();
optimizer.step();
```

This slice is incomplete until:

- values, shapes, strides, dtypes, aliasing, and gradients match the pinned PyTorch oracle;
- a saved-tensor mutation/version-counter regression test exists;
- the CPU reference backend and meta backend both implement the operators;
- a tiny executable example runs from a clean checkout;
- the TensorLab demo performs one deterministic training step using the same runtime;
- Linux, Windows, and macOS CI build it.

### 0.3 Non-negotiable execution behavior

- Do not claim parity from API surface alone.
- Do not add hundreds of unsupported stubs.
- Do not hide missing semantics behind CPU copies or dtype coercions.
- Do not bypass the dispatcher from public APIs.
- Do not add preview JDK APIs to release artifacts.
- Do not merge code that only works on the agent's platform.
- Do not make demo applications separate toy codepaths; they must consume published JTorch modules exactly as external applications would.

## 1. Mission

JTorch is a **JDK 25-native, Java-first, PyTorch-compatible tensor and deep-learning platform** for research, training, inference, compilation, and deployment across Linux, Windows, and macOS.

“JDK 25-native” means:

- Java 25 class-file target and runtime baseline;
- stable Foreign Function & Memory API for native interoperability and off-heap host memory;
- `MemorySegment`, `Arena`, `Linker`, and `SymbolLookup` as the canonical native-memory/native-call substrate;
- finalized `ScopedValue` for dynamically scoped execution state;
- virtual threads for I/O and orchestration, not numerical compute scheduling;
- the standard Class-File API for generated and transformed JVM bytecode;
- JPMS module boundaries and explicit native-access declarations;
- no JNI implementation path in JTorch-owned code;
- no preview features in release artifacts;
- the incubating Vector API isolated behind an optional implementation module and absent from public signatures.

The long-term product must provide:

- eager tensor computation with PyTorch-compatible dense, sparse, nested, quantized, and meta semantics;
- dynamic reverse-mode and forward-mode automatic differentiation;
- function transforms including VJP, JVP, Jacobian, Hessian, vectorization, and functionalization;
- neural-network modules, losses, optimizers, schedulers, datasets, data loaders, and checkpointing;
- CPU, CUDA, ROCm, and extensible custom-device execution;
- automatic mixed precision, quantization, sparsity, and structured tensor layouts;
- graph capture, symbolic shapes, compilation, export, deployable runtime packages, and generated CPU kernels;
- distributed collectives, DDP, sharded training, distributed tensors, and fault-aware checkpointing;
- safe model and weight interoperability;
- a compatibility program continuously validated against pinned PyTorch releases;
- real cross-platform applications that prove end-to-end training and inference rather than isolated microbenchmarks;
- an idiomatic Java API and optional compatibility frontends for Kotlin, Scala, and Python.

JTorch is not a wrapper whose main implementation delegates operations to Python or a running PyTorch process. The runtime, dispatcher, tensor metadata, autograd engine, operator registry, module system, memory model, compiler IR, packaging, and demo applications belong to JTorch.

Native vendor libraries and JTorch-owned native kernels are allowed in optional performance backends. “JVM-native” does not mean rewriting cuBLAS or cuDNN in Java; it means the JVM owns execution and interoperates through stable, explicit FFM boundaries.

## 2. What “drop-in replacement for PyTorch” means

The phrase is ambiguous. JTorch must report compatibility in explicit tiers and must never claim a tier that is not continuously tested.

### Tier A — Operator semantic compatibility

For covered operators, JTorch must match the pinned PyTorch oracle for:

- signatures and overload behavior;
- shape rules and symbolic-shape constraints;
- dtype inference and promotion;
- broadcasting;
- device and layout rules;
- aliasing, views, mutation, in-place operations, and `out` variants;
- forward values;
- reverse- and forward-mode derivatives;
- errors by category and cause;
- random-generator state semantics;
- transform behavior under autograd, functionalization, batching/vectorization, autocast, fake/meta execution, and graph capture where applicable.

### Tier B — Model and weight compatibility

JTorch must load and execute model state through safe, versioned adapters:

- Safetensors;
- ONNX where the operator set is supported;
- PyTorch exported-program/PT2 adapters for explicitly supported versions;
- restricted, non-executing import of well-defined PyTorch state-dict subsets;
- optional GGUF for inference-focused ecosystems.

Arbitrary Python pickle execution is not an acceptable model-loading mechanism.

### Tier C — Java source-level familiarity

A PyTorch user should be able to port a model to Java mechanically. Names, defaults, module composition, state dictionaries, training/evaluation mode, hooks, optimizer parameter groups, and tensor semantics should be recognizable without forcing un-Java-like APIs.

This is not identical Python syntax. The canonical Java facade is `Torch`, not a lowercase Java class.

### Tier D — Exported-program compatibility

JTorch must import, validate, transform, and execute supported exported graphs with:

- explicit parameters and buffers;
- symbolic dimensions and range constraints;
- a stable JTorch operator-set version;
- dialect/version adapters;
- clear rejection of unsupported effects or operators.

### Tier E — Python source compatibility

Literal “run existing Python code after replacing `import torch`” requires an optional Python facade. A Java API alone cannot satisfy it.

The facade is a separate product track and may use GraalPy, a CPython native bridge, or another versioned mechanism chosen by ADR. It must delegate to the JTorch runtime and share the same operator schemas and compatibility suite. JTorch must be described publicly as “PyTorch-compatible for the JVM” until unmodified Python compatibility reaches a measured release gate.

### Compatibility ledger

The repository must contain machine-readable coverage:

```text
compat/
├── baseline.toml
├── operators.json
├── modules.json
├── optimizers.json
├── transforms.json
├── distributed.json
├── formats.json
└── reports/
```

Every entry must be one of:

- `unsupported`;
- `schema-only`;
- `forward`;
- `autograd`;
- `full-semantics`;
- `compiled`;
- `distributed`;
- `deprecated`.

No marketing percentage may count `schema-only` as implemented.

---

## 3. Hard constraints

### 3.1 JDK baseline

- All production Java artifacts require JDK 25 or newer and target Java 25 class files.
- Gradle must run on JDK 25. Use Gradle 9.1 or newer; pin the exact wrapper version.
- Public APIs may use finalized Java 25 language and library features.
- Release builds must not require `--enable-preview`.
- Structured Concurrency, Stable Values, primitive pattern previews, and other preview APIs are forbidden in stable source sets.
- The Vector API remains incubating in JDK 25 and must live only in `jtorch-cpu-vector`; it must never appear in public API signatures.
- Linux, Windows, and macOS are first-class targets. Platform-specific behavior requires conformance tests and capability reporting.
- JPMS module descriptors are required for published artifacts. Automatic-module-name-only publishing is insufficient for 1.0.

### 3.2 Native interoperability

- The stable Foreign Function & Memory API is the sole JTorch-owned native interoperability mechanism.
- JNI declarations, `JNIEnv*` glue, generated JNI headers, and Java array pinning are forbidden.
- Native libraries expose C ABI entry points. Never bind directly to unstable C++ ABI symbols.
- FFM downcalls/upcalls, layouts, symbols, ownership, and error translation are generated or centrally defined and tested.
- Launchers and packaged demos must supply `--enable-native-access` for only the modules that require it.
- Native access is capability-based: the CPU reference runtime must function without loading a native library.
- Native library loading must verify platform, architecture, expected ABI version, and optional checksum/signature metadata.

### 3.3 Memory model

- `MemorySegment` is the canonical representation for off-heap and mapped host storage.
- `Arena` ownership must be explicit and never tied to a short-lived lexical scope when tensors can escape.
- User-facing `Tensor` is not generally `AutoCloseable`; storage handles and allocators manage shared lifetimes.
- No finalizer-based correctness. Cleaner-based diagnostics may be used only as a leak warning/failsafe.
- Host, mapped, pinned, unified, and device memory have distinct storage implementations and synchronization rules.
- Large tensor storage must not depend on Java array indexing limits.
- Every native allocation path participates in leak accounting and stress tests.

### 3.4 Dependency policy

Zero third-party **JVM runtime** dependencies are required for:

- `jtorch-api`;
- `jtorch-core`;
- `jtorch-schema` runtime tables;
- `jtorch-dispatch`;
- `jtorch-tensor`;
- `jtorch-autograd`;
- `jtorch-cpu-reference`;
- core NN/optimizer/data APIs;
- the native JTorch checkpoint reader/writer;
- `jtorch-demo-common` and the standard desktop demo shells.

Allowed dependencies:

- Gradle and build plugins used only during build, lint, test, package, or publication;
- JUnit and bounded property/fuzzing tools in test scope;
- Python and a pinned PyTorch environment used only as the differential oracle, compatibility extractor, and format converter;
- optional vendor-native runtimes such as CUDA, cuBLAS, cuDNN, NCCL, ROCm/HIP, oneDNN, OpenCL, and Vulkan;
- optional interop adapters whose dependency graph is explicit and does not leak into core;
- model files, tokenizers, and datasets downloaded through checksummed manifests with documented licenses.

“No external dependencies” means core JTorch users do not inherit a Java dependency graph. It does not mean pretending a production GPU backend can exist without vendor drivers and compute libraries.

### 3.5 Correctness before speed

- The Java CPU reference backend is JTorch’s internal semantic reference.
- A pinned PyTorch environment is the external behavioral oracle.
- Optimized kernels must be switchable to reference kernels at runtime.
- Every optimization requires differential and alias/autograd coverage.
- Tolerance changes require recorded numerical evidence.
- Determinism claims are per operator/backend/algorithm, never global marketing statements.

### 3.6 No false completeness

- No empty method bodies, placeholder returns, fake metrics, or silent exceptions.
- No fake GPU backend that copies to CPU unless explicitly selected as a diagnostic fallback and clearly reported.
- Unsupported behavior throws a typed exception containing operator, overload, inputs, dispatch keys, backend, and compatibility status.
- Generated API symbols remain marked unsupported in the machine-readable ledger until semantic tests pass.
- Demo applications must not substitute canned output for model execution.
- A demo is not “working” until its model path, input path, result path, cancellation, and error handling are exercised.

### 3.7 Extensibility

- Backends, operators, model loaders, profiler sinks, and demo model providers use versioned extension contracts.
- Use JPMS services/`ServiceLoader` for zero-dependency provider discovery.
- Reflection is forbidden in numerical hot paths and optional elsewhere.
- Public extension contracts must not be sealed against third-party modules.

## 4. Product boundaries

### 4.1 Core framework

Core includes tensor semantics, schemas, dispatcher, operators, autograd, modules, optimizers, data APIs, serialization foundations, compile/export foundations, distributed abstractions, profiling hooks, and the CPU reference backend.

### 4.2 Optional backend artifacts

- optimized pure-Java CPU;
- incubating Vector API CPU implementation;
- FFM-native CPU libraries such as BLAS/oneDNN;
- CUDA;
- ROCm/HIP;
- Metal/MPS research backend;
- Vulkan/OpenCL research backends;
- XPU/oneAPI;
- remote/custom devices.

Do not build every backend before tensor semantics, dispatch, and CPU reference behavior are stable. Priority is CPU reference, optimized CPU, CUDA, then ROCm. Other backends are accepted only with the common conformance suite.

### 4.3 Ecosystem modules

Vision, audio, tokenizers, transformer architectures, diffusion pipelines, model manifests, and application-oriented preprocessing live outside the stable tensor core. They validate framework breadth without forcing model-specific assumptions into tensor abstractions.

### 4.4 Demo applications

Demo applications are product-quality consumers inside the monorepo. They must:

- depend on JTorch through published module boundaries;
- use the same serialization, tokenization, dispatch, and backend selection paths as external users;
- build on Linux, Windows, and macOS;
- support a deterministic headless smoke mode;
- provide actionable, real-world behavior;
- include accessibility, cancellation, progress, diagnostics, and safe model acquisition;
- never become privileged internal APIs unavailable to third-party applications.

### 4.5 AI optimization family

Automatic mixed precision, quantization, sparsity, pruning, and deployment transformations form a separable `jtorch-ao` family. They may evolve faster than core but must preserve operator and serialization contracts.

## 5. Repository topology

Use a Gradle 9.x multi-project build with Kotlin DSL and a checked-in wrapper. `AGENTS.md` is the design source of truth; do not duplicate its architecture into multiple drifting Markdown files.

```text
jtorch/
├── AGENTS.md
├── README.md
├── LICENSE
├── NOTICE
├── settings.gradle.kts
├── build.gradle.kts
├── gradle.properties
├── gradlew
├── gradlew.bat
├── gradle/
│   ├── wrapper/
│   ├── libs.versions.toml
│   └── verification-metadata.xml
├── build-logic/
│   └── src/main/kotlin/
├── .github/
│   ├── actions/setup-jtorch-build/
│   └── workflows/
│       ├── ci.yml
│       ├── demos.yml
│       ├── oracle.yml
│       ├── native.yml
│       ├── gpu.yml
│       ├── nightly.yml
│       ├── codeql.yml
│       └── release.yml
├── compat/
│   ├── baseline.toml
│   ├── operators.json
│   ├── modules.json
│   ├── optimizers.json
│   ├── transforms.json
│   ├── distributed.json
│   ├── formats.json
│   └── reports/
├── schema/
│   ├── ops/
│   ├── derivatives/
│   ├── decompositions/
│   ├── modules/
│   └── versioning/
├── tools/
│   ├── opgen/
│   ├── oracle/
│   ├── compatibility-report/
│   ├── ffm-bindings/
│   ├── checkpoint-converter/
│   ├── model-manifest/
│   └── release/
├── modules/
│   ├── jtorch-api/
│   ├── jtorch-core/
│   ├── jtorch-schema/
│   ├── jtorch-dispatch/
│   ├── jtorch-tensor/
│   ├── jtorch-memory/
│   ├── jtorch-autograd/
│   ├── jtorch-func/
│   ├── jtorch-ops-prims/
│   ├── jtorch-ops-aten/
│   ├── jtorch-nn/
│   ├── jtorch-optim/
│   ├── jtorch-data/
│   ├── jtorch-serialization/
│   ├── jtorch-compile/
│   ├── jtorch-export/
│   ├── jtorch-distributed/
│   ├── jtorch-profiler/
│   └── jtorch-library/
├── backends/
│   ├── jtorch-cpu-reference/
│   ├── jtorch-cpu-optimized/
│   ├── jtorch-cpu-vector/
│   ├── jtorch-native-ffm/
│   ├── jtorch-cuda/
│   ├── jtorch-rocm/
│   ├── jtorch-metal/
│   ├── jtorch-vulkan/
│   └── jtorch-opencl/
├── interop/
│   ├── jtorch-safetensors/
│   ├── jtorch-onnx/
│   ├── jtorch-pytorch-export/
│   ├── jtorch-gguf/
│   └── jtorch-python/
├── ao/
│   ├── jtorch-amp/
│   ├── jtorch-quantization/
│   └── jtorch-sparsity/
├── ecosystem/
│   ├── jtorch-vision/
│   ├── jtorch-audio/
│   ├── jtorch-tokenizers/
│   ├── jtorch-transformers/
│   └── jtorch-diffusion/
├── demos/
│   ├── jtorch-demo-common/
│   ├── tensorlab/
│   ├── vision-inspector/
│   ├── semantic-search/
│   ├── gc-doctor/
│   ├── transcribe/
│   ├── local-chat/
│   └── image-studio/
├── model-manifests/
│   ├── fixtures/
│   ├── demos/
│   └── licenses/
├── examples/
├── benchmarks/
└── tests/
    ├── unit/
    ├── parity/
    ├── autograd/
    ├── transforms/
    ├── aliasing/
    ├── compiler/
    ├── distributed/
    ├── serialization/
    ├── demos/
    ├── models/
    ├── stress/
    └── performance/
```

During early milestones, closely coupled internals may share a physical Gradle project to reduce build friction. Public packages and dependency direction must still follow this design.

### 5.1 Dependency direction

```text
api ← core ← tensor/memory ← dispatch ← backend implementations
              ↑                ↑
           autograd       schema/generated registries
              ↑
   nn / optim / func / compile / distributed / ecosystem
              ↑
         demos and external applications
```

Rules:

- Backends depend on core contracts; core never depends on a concrete accelerator.
- `jtorch-memory` may use stable FFM but must not depend on CUDA or model modules.
- Autograd invokes operators through the dispatcher, never direct CPU kernels.
- Compiler IR depends on schemas, not Java wrapper method bodies.
- Ecosystem modules do not add dependencies to core.
- Demos depend only on public JTorch artifacts and `jtorch-demo-common`.
- Demo source sets may not import `*.internal.*` packages.
- Interop adapters are versioned at their boundaries.
- The optional Vector API module is never required by core or demos.

### 5.2 Build conventions

Every Java project applies shared conventions that enforce:

```kotlin
java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
    modularity.inferModulePath.set(true)
    withSourcesJar()
    withJavadocJar()
}

tasks.withType<JavaCompile>().configureEach {
    options.release.set(25)
    options.encoding = "UTF-8"
    options.compilerArgs.addAll(listOf("-Xlint:all", "-Werror"))
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
    jvmArgs("--enable-native-access=org.jtorch.nativeffm")
}
```

Do not globally enable native access in production launchers. Each packaged application declares the precise JTorch modules that require it.

## 6. Canonical schema and code generation

Modern framework parity is impossible if thousands of overloads, derivatives, dispatch registrations, docs, and tests are handwritten independently.

JTorch must be metadata-driven.

### 6.1 Canonical JTorch operator schema

Each operator record must describe:

- namespace and base name;
- overload name;
- arguments, defaults, keyword-only markers, optionals, lists, tuples, scalars, generators, devices, layouts, dtypes, and memory formats;
- return structure;
- alias sets;
- mutability and write effects;
- view behavior;
- functional, in-place, and `out` relationships;
- dtype promotion policy;
- device selection policy;
- shape/meta function;
- differentiability;
- reverse- and forward-mode derivative registration;
- decomposition;
- transform rules;
- backend kernels and fallbacks;
- determinism classification;
- compatibility status and oracle version;
- documentation and examples.

Illustrative internal schema:

```yaml
name: aten::add.Tensor
arguments:
  - { name: self, type: Tensor, alias: a }
  - { name: other, type: Tensor, alias: b }
  - { name: alpha, type: Scalar, default: 1, keyword_only: true }
returns:
  - { type: Tensor, alias: null }
variants: [function, method, out, inplace]
promotion: elementwise_default
meta: add_meta
reverse_ad: add_backward
forward_ad: add_jvp
decomposition: null
transforms: [functionalize, vmap, autocast, fake]
```

### 6.2 Generated outputs

`tools/opgen` generates:

- Java facade overloads;
- `Tensor` methods;
- operator identifiers and schemas;
- dispatcher registration tables;
- backend kernel interfaces;
- meta/fake registration tables;
- autograd node scaffolding;
- decomposition registration;
- API documentation tables;
- parity-test case descriptors;
- compatibility reports.

Generated files must carry a header and must never be edited manually.

### 6.3 PyTorch compatibility extraction

Maintain a pinned external PyTorch source/runtime as an oracle. Extract and normalize:

- public schemas;
- ATen/native operator metadata;
- derivative metadata;
- decomposition metadata;
- dtype-promotion observations;
- module and optimizer signatures;
- serialization/export version metadata.

Do not blindly copy implementation code. Treat PyTorch metadata as compatibility input, preserve required attribution, and review licensing in an ADR and `NOTICE`.

The checked-in JTorch schema is the build input. Oracle extraction updates it through reviewable diffs; ordinary builds must not require network access or Python.

---

## 7. Runtime value model

Operator schemas require more than tensors. Define an internal immutable value algebra.

```text
Value
├── NoneValue
├── TensorValue
├── ScalarValue
├── IntValue / SymIntValue
├── FloatValue / SymFloatValue
├── BoolValue / SymBoolValue
├── StringValue
├── DeviceValue
├── DTypeValue
├── LayoutValue
├── MemoryFormatValue
├── GeneratorValue
├── ListValue<T>
├── TupleValue
├── DictValue<K,V>
└── CustomValue
```

Requirements:

- no raw `Object[]` in runtime contracts;
- preserve optionality and tuple/list distinction;
- support multiple outputs and non-tensor outputs;
- support symbolic values in meta and compiler modes;
- provide generated typed wrappers at public API boundaries;
- permit custom values only through versioned registration.

---

## 8. Tensor, storage, layouts, and aliasing

### 8.1 Tensor handle

A public `Tensor` is a lightweight handle to a `TensorImpl`. It is not a Java record and is not `AutoCloseable` in the ordinary API.

```java
public class Tensor {
    final TensorImpl impl;

    public Shape shape();
    public Strides strides();
    public DType dtype();
    public Device device();
    public Layout layout();
    public boolean requiresGrad();
    public Tensor grad();
    public long numel();
    public boolean isContiguous();
}
```

Why `Tensor` is not normally `AutoCloseable`:

- views share storage;
- autograd may retain saved tensors;
- parameters and buffers have module-owned lifetimes;
- requiring try-with-resources for ordinary tensor expressions is hostile and unlike PyTorch.

Native memory uses reference-counted storage handles plus backend allocators. An optional `TensorScope` can deterministically release temporary native tensors in services and benchmarks without changing default semantics.

### 8.2 TensorImpl

Internal fields include:

- storage handle;
- storage offset;
- sizes;
- strides;
- dtype;
- device;
- layout;
- memory format;
- dispatch-key set;
- version counter;
- autograd metadata;
- view metadata and replay function;
- names, conjugate/negative bits, and other semantics only when implemented.

### 8.3 Shape and strides

Do not use a Java record containing a mutable array without custom equality and defensive copying.

`Shape` and `Strides` must be immutable value classes with:

- defensive construction;
- value-based equality/hash code;
- checked `numel` overflow;
- negative-dimension normalization where allowed;
- symbolic dimensions in compiler/meta contexts through a separate `SymShape` representation.

### 8.4 Storage

```text
Storage
├── HeapStorage                  small/testing-only primitive arrays
├── NativeCpuStorage             aligned MemorySegment allocation
├── MappedStorage                file-backed MemorySegment
├── PinnedHostStorage            backend-managed transfer memory
├── SharedMemoryStorage          IPC/distributed host region
├── CudaStorage                  device allocation handle
├── RocmStorage                  device allocation handle
├── MetalStorage                 device allocation handle
└── CustomStorage
```

Core interfaces:

```java
public interface Storage {
    long byteSize();
    Device device();
    StorageKind kind();
    StorageHandle handle();
    boolean isReadOnly();
}

public interface HostAccessibleStorage extends Storage {
    MemorySegment segment();
}
```

`StorageHandle` owns shared lifetime and allocator provenance. A storage implementation may use an `Arena`, native backend handle, or device pointer internally, but public tensor code never closes it directly.

Storage requirements:

- byte length, address-range, endianness, and dtype-alignment validation;
- reference-counted shared ownership independent from Java object reachability;
- deterministic release when the last owning handle is retired and asynchronous work is complete;
- allocator provenance, stream/event ownership, and debug allocation traces;
- versioned access and mutation tracking;
- safe zero-size allocations and zero-length mapped regions;
- support for tensors larger than Java array limits;
- leak diagnostics in debug/test mode;
- no use-after-free when views, autograd saved tensors, or queued device kernels outlive a wrapper;
- mapped-file bounds and truncation protection;
- no assumption that a mapped segment is physically zero-copy on every operating system.

An illustrative host allocator:

```java
final class NativeCpuAllocation implements AutoCloseable {
    private final Arena arena;
    private final MemorySegment segment;
    private final AtomicBoolean closed = new AtomicBoolean();

    NativeCpuAllocation(long bytes, long alignment) {
        if (bytes < 0 || alignment <= 0 || Long.bitCount(alignment) != 1) {
            throw new IllegalArgumentException("Invalid allocation request");
        }
        this.arena = Arena.ofShared();
        this.segment = arena.allocate(bytes, alignment);
    }

    MemorySegment segment() {
        if (closed.get()) throw new IllegalStateException("Allocation released");
        return segment;
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) arena.close();
    }
}
```

The actual allocator uses pooled slabs and `StorageHandle`; the example only demonstrates explicit FFM ownership.

### 8.5 Layouts

Final design must accommodate:

- strided dense;
- sparse COO;
- sparse CSR/CSC;
- sparse BSR/BSC;
- nested/jagged;
- quantized;
- meta/fake;
- backend-private layouts.

Do not encode all layouts as a dense tensor plus flags. Layout-specific metadata belongs in specialized `TensorImpl`/layout components while preserving the public `Tensor` handle.

### 8.6 Views, mutation, and version counters

Must implement:

- storage alias sets;
- view replay metadata;
- base/view relationship;
- version counters shared by aliases;
- in-place legality checks;
- saved-tensor version validation during backward;
- `detach` and detached-view semantics;
- `copy_`, `set_`, resize, and `out` semantics only when tests define them;
- functionalization that can replace mutations and views with functional equivalents for transforms and compilation.

Aliasing is a first-class semantic, not an optimization detail.

---

## 9. Dispatcher

A single `switch(device)` is insufficient. JTorch needs a composable dispatcher because the same operator can be intercepted by autograd, transforms, autocast, fake tensors, custom backends, and compiler modes.

Dynamically scoped dispatcher, grad mode, inference mode, autocast, current device, current stream, forward-AD level, and capture context use finalized `ScopedValue` where lexical scoping fits. Mutable per-stream/backend objects remain explicit handles. Do not recreate a web of inheritable thread locals.

### 9.1 Dispatch keys

Implement a prioritized `DispatchKeySet`. Initial and final keys include:

```text
PreDispatch / UserInterception
Functionalize
PythonFacade (optional)
Autograd
ForwardAD
Vmap/Batching
AutocastCPU / AutocastCUDA
BackendSelect
Meta/Fake
Sparse / Quantized / Nested layout keys
CPU
CUDA
ROCM
Metal
Vulkan/OpenCL
CompositeExplicitAutograd
CompositeImplicitAutograd
Fallback
```

Exact ordering is an ADR-backed runtime contract and must have dispatcher-table tests.

### 9.2 Registration

Equivalent concepts to provide:

- operator schema definition;
- implementation registration by namespace, overload, and dispatch key;
- fallthrough registrations;
- backend fallback;
- composite implementation;
- custom operators;
- fake/meta implementation;
- autograd registration;
- vmap and autocast rules.

Expose an extension API in `jtorch-library` analogous in purpose to modern custom-operator registration.

### 9.3 Redispatch

Wrappers such as autograd or autocast must remove their own dispatch key and redispatch to lower layers. Direct backend calls are forbidden outside backend tests.

### 9.4 Meta execution

Every core operator needs a meta function that computes output shape, dtype, layout, aliasing, and constraints without allocating real data. Meta is used by:

- schema validation;
- fake tensors;
- export;
- compiler shape propagation;
- memory planning;
- generated tests.

### 9.5 Composite operators and decompositions

Operators expressible from primitives should prefer registered decompositions. Distinguish:

- composite implementations that remain differentiable through primitive ops;
- explicit-autograd implementations with custom backward;
- backend kernels;
- compiler-only decompositions.

Decomposition changes are semantic changes and require parity tests.

---

## 10. Operator semantics

### 10.1 Operator families

Final stable coverage includes:

- creation and factories;
- indexing, slicing, gather/scatter, masking;
- elementwise arithmetic, comparison, logical, bitwise, and special functions;
- reductions and statistics;
- sorting, selection, and uniqueness;
- shape, view, concatenation, split, and padding;
- linear algebra and matrix decompositions;
- FFT;
- random distributions;
- convolution, pooling, normalization, activation, losses;
- attention and transformer primitives;
- sparse and nested operations;
- quantized operations;
- distributed tensor operations;
- utility and device operations.

### 10.2 Dtype system

Support, subject to backend capability:

- bool;
- signed and unsigned integers where compatible;
- float16, bfloat16, float32, float64;
- complex64, complex128;
- float8 families in AO/accelerator modules;
- packed int4/uint4 and other sub-byte types as quantized/storage dtypes, not ordinary Java primitives.

Promotion tables are generated from explicit rules and differential tests. Never encode illustrative promotion examples as unreviewed truth.

### 10.3 Scalar semantics

A `Scalar` value must preserve integer, floating, complex, boolean, and zero-dimensional-tensor distinctions where operator semantics require them.

### 10.4 Memory formats

Support contiguous and channels-last formats and preserve format according to operator semantics. `contiguous()` is not a universal hidden fix for kernel limitations; fallbacks must be explicit and tested.

### 10.5 Structured kernels

Where applicable, separate:

1. meta/out-shape logic;
2. output allocation or `out` validation;
3. backend implementation.

This reduces divergence between functional, in-place, and `out` variants.

---

## 11. Automatic differentiation and function transforms

### 11.1 Reverse-mode autograd

Core concepts:

```text
AutogradMeta
Node
Edge
GradientEdge
SavedTensor
AccumulateGrad
GraphRoot
GraphTask
ReadyQueue
Engine
Hook
```

Requirements:

- dynamic tape construction;
- leaf/non-leaf semantics;
- gradient accumulation;
- `backward` and `grad` APIs;
- multiple outputs and inputs;
- retain/create graph;
- higher-order gradients;
- reentrant backward;
- hooks on tensors, nodes, modules, and saved tensors;
- anomaly detection;
- no-grad and inference modes;
- checkpointing with RNG preservation policies;
- custom autograd functions;
- thread-safe graph execution;
- asynchronous device synchronization that avoids global synchronization.

### 11.2 Saved tensors

Saved tensors store handles plus required metadata and expected version counters. Backward must fail diagnostically when an in-place mutation invalidates a saved value.

### 11.3 Forward-mode AD

Provide dual tensors/levels and JVP propagation. Forward AD is not a post-1.0 afterthought because composable transforms and higher-order derivatives depend on it.

### 11.4 Function transforms (`jtorch.func`)

Final transforms:

- `grad` and `gradAndValue`;
- `vjp`;
- `jvp`;
- `jacrev` and `jacfwd`;
- `hessian`;
- `vmap`;
- `functionalize`;
- functional calls over modules and explicit parameter/buffer maps.

Transforms must compose. Operators declare batching, randomness, mutation, and AD behavior through dispatcher registrations or decompositions.

### 11.5 Custom functions

A custom autograd function must define forward/setup-context and transformation-compatible backward/JVP/vmap rules where claimed. Custom functions that use opaque native code must provide fake/meta behavior for compile/export support.

---

## 12. Randomness and determinism

Provide explicit `Generator` objects and a default generator per device.

Requirements:

- counter-based generator suitable for parallel CPU and GPU kernels;
- stable state serialization per JTorch release policy;
- manual seeding;
- fork/split behavior for workers and distributed ranks;
- reproducible dropout and sampling when deterministic mode permits;
- deterministic-algorithm policy that rejects known nondeterministic kernels when enabled;
- checkpointing integration;
- statistical tests for distributions rather than only fixed-output tests.

Do not promise bit-for-bit equality with PyTorch across all devices. Promise documented semantic and statistical compatibility, and exact reproducibility only within defined JTorch configurations.

---

## 13. Neural-network module system

### 13.1 Module registry

`Module` explicitly registers:

- parameters;
- persistent and non-persistent buffers;
- child modules;
- forward/pre-forward/backward hooks;
- training/evaluation mode.

Do not rely on reflection as the only discovery mechanism.

```java
public abstract class Module {
    protected final <M extends Module> M registerModule(String name, M module);
    protected final Parameter registerParameter(String name, Parameter parameter);
    protected final Tensor registerBuffer(String name, Tensor tensor, boolean persistent);

    public final StateDict stateDict();
    public final LoadStateResult loadStateDict(StateDict state, LoadStateOptions options);
    public final Module train(boolean training);
}
```

Generated or typed adapters may provide ergonomic `forward` signatures. Internally, general calls use the `Value` algebra so modules can accept and return nested structures.

### 13.2 Parameter

`Parameter` is a registered trainable tensor role with stable identity and name. Whether implemented as a Tensor subtype or wrapper must be resolved by ADR before public API freeze; interoperability and module traversal matter more than superficial inheritance.

### 13.3 State dictionaries

State dictionaries are ordered name-to-tensor mappings with metadata. They must preserve:

- parameters and persistent buffers;
- tied/shared storage relationships where supported;
- module version metadata;
- strict and non-strict loading;
- missing/unexpected keys;
- dtype/device mapping;
- sharded state in distributed modules.

### 13.4 Required modules

Final core NN coverage includes containers, linear/bilinear, embeddings, convolutions, transposed convolutions, pooling, padding, normalization, recurrent layers, attention/transformers, dropout, activations, losses, initialization, parametrizations, pruning hooks, and lazy modules where semantics are defined.

---

## 14. Optimizers and schedulers

Optimizer design must support:

- parameter groups;
- per-group defaults;
- state keyed by stable parameter identity;
- differentiable/capturable/fused modes where supported;
- foreach-style batched updates;
- zero-grad modes;
- optimizer hooks;
- state-dict round trips;
- mixed-precision scaler interaction;
- distributed/sharded optimizer state.

Initial optimizers:

- SGD;
- Adam;
- AdamW;
- RMSprop;
- Adagrad;
- Adadelta;
- RAdam;
- NAdam;
- ASGD;
- LBFGS.

Experimental algorithms belong outside parity claims unless present in the selected PyTorch baseline.

Schedulers must cover the stable PyTorch families and preserve step/order semantics.

---

## 15. Data pipeline

Provide:

- map-style and iterable datasets;
- samplers and batch samplers;
- collate functions;
- deterministic shuffling and generator injection;
- virtual-thread worker orchestration for blocking I/O, plus bounded platform-thread/process workers for CPU-heavy transforms;
- prefetching;
- persistent workers;
- pinned-memory handoff where a backend supports it;
- distributed samplers;
- cancellation, timeout, and worker-failure propagation;
- backpressure for iterable sources.

The first implementation may use virtual threads for blocking I/O and bounded platform threads for numerical transforms. A process-worker mode is required for isolation and native-library safety but must use an explicit IPC protocol rather than Java object serialization.

---

## 16. Serialization and interoperability

### 16.1 Native JTorch format

Define a safe, versioned format with:

- manifest;
- schema/opset version;
- tensors stored in bounded binary segments;
- checksums;
- storage-sharing metadata;
- state-dict metadata;
- optional graph IR;
- no executable object deserialization;
- streaming and memory-mapped reads;
- size and allocation limits.

Document forward/backward compatibility. Add migration tools before changing the stable format.

### 16.2 Safetensors

Support load and save with:

- strict bounds checking;
- duplicate/overlap rejection;
- dtype and endianness validation;
- memory mapping when alignment and backend permit;
- explicit copies/dequantization when zero-copy is impossible.

Never claim zero-copy merely because a file is memory mapped.

### 16.3 PyTorch checkpoints

PyTorch checkpoint files may involve Python pickle and executable reconstruction. JTorch must not execute arbitrary pickle payloads.

Supported paths:

1. preferred: a provided conversion tool running in a sandboxed pinned Python environment writes Safetensors or native JTorch state;
2. restricted reader: only a documented non-executing subset with strict opcode and type allowlists;
3. exported program adapter: versioned PT2 archive support where the format and opset are understood.

### 16.4 ONNX

Import/export is opset-versioned. Preserve symbolic shapes, initializers, control flow, optional values, and external tensor data. Unsupported operators must be reported before execution.

### 16.5 GGUF

GGUF is optional and inference-oriented. Keep it out of core training semantics. Support mapped quantized weights and architecture adapters without making GGUF the canonical JTorch model format.

---

## 17. Compiler, capture, and export

JTorch must have an eager-first architecture with an optional compiler stack.

### 17.1 Modes

- eager;
- fake/meta;
- trace/capture;
- exported program;
- compiled execution;
- ahead-of-time package.

### 17.2 Capture reality on Java

PyTorch’s Python bytecode capture mechanism cannot be copied directly. JTorch initial capture uses operator interception with fake/proxy tensors and explicit structured-control-flow operators.

Data-dependent Java control flow that consumes actual tensor values causes a graph break or export failure unless expressed through supported control-flow primitives. Optional bytecode instrumentation may be researched in a separate module and ADR; it is not a core assumption.

### 17.3 IR layers

```text
Eager calls
  ↓ capture
JTorch Graph IR (user-visible call structure)
  ↓ functionalization and normalization
Functional ATen-compatible IR
  ↓ decomposition
Primitive IR
  ↓ shape specialization / guards / optimization
Backend IR
  ↓ scheduling and memory planning
Kernel calls or generated code
```

IR values support tensors, symbolic integers/floats/bools, lists, tuples, optionals, tokens, and effects.

### 17.4 Symbolic shapes

Implement `SymInt`, `SymFloat`, `SymBool`, range constraints, equality constraints, guards, and shape environments. Avoid baking example sizes into exported graphs without declaring them static.

### 17.5 Effects

Model mutation, RNG, I/O/custom calls, collectives, and asynchronous device work as explicit effects/tokens where required. Functionalization removes ordinary tensor mutation but cannot erase external effects.

### 17.6 Required passes

- validation and canonicalization;
- functionalization;
- decomposition;
- constant propagation/folding;
- dead-code elimination;
- common-subexpression elimination where alias-safe;
- shape propagation and guard simplification;
- dtype/layout propagation;
- fusion grouping;
- device placement;
- liveness and memory planning;
- scheduling;
- backend lowering.

### 17.7 AOT autograd

Training compilation creates functional forward and backward graphs with saved values and mutation handling. Compiled and eager gradients must share differential tests.

### 17.8 Exported program

A stable JTorch exported program contains:

- graph;
- input/output tree specification;
- parameters, buffers, constants;
- graph signature;
- symbolic constraints;
- module-call metadata;
- operator-set versions;
- custom-op requirements;
- verifier results.

PyTorch PT2 import is an adapter into this model, not the internal format itself.

---

## 18. Backend architecture

### 18.1 Backend contract

Backends provide:

- device discovery and immutable properties;
- allocator, stream/queue, event, and synchronization primitives;
- host/device and peer data transfer;
- kernel registration and capability negotiation;
- error translation with deferred asynchronous error checks;
- RNG state and deterministic-algorithm metadata;
- profiler hooks;
- native ABI version reporting;
- architecture and library version reporting.

Operator execution returns a general `Value` result and supports preallocated outputs and tuples, not only one tensor.

### 18.2 CPU reference backend

- pure finalized Java 25 APIs;
- `MemorySegment` and checked layouts for native host storage;
- straightforward deterministic kernels where practical;
- arbitrary strided semantics before fast contiguous paths;
- no native library required;
- no incubating Vector API;
- correctness, diagnostics, and cross-platform reproducibility prioritized over speed.

### 18.3 Optimized Java CPU backend

Optimization ladder:

1. cache-aware and tiled Java kernels over `MemorySegment`;
2. carefully bounded platform-thread compute pools;
3. generated dtype/rank/contiguity-specialized kernels using the standard Class-File API;
4. optional `jtorch-cpu-vector` implementations;
5. optional FFM-native BLAS/oneDNN kernels;
6. graph-level fusion and memory planning.

Virtual threads are forbidden as a substitute for compute scheduling. They may coordinate blocking model loading, preprocessing, and asynchronous service requests.

### 18.4 Vector API module

JDK 25’s Vector API remains incubating. Therefore:

- `jtorch-cpu-vector` is optional;
- it compiles and runs with `--add-modules jdk.incubator.vector`;
- no public JTorch signature references vector classes;
- CI has a dedicated job that may fail independently before the module is declared stable;
- releases remain fully functional without it;
- scalar/reference kernels are always available for correctness fallback.

### 18.5 FFM native binding layer

`jtorch-native-ffm` centralizes:

- native library discovery and explicit loading;
- `Linker` and `SymbolLookup` management;
- generated `FunctionDescriptor`, `MemoryLayout`, and downcall handles;
- ABI/version negotiation;
- platform-specific naming and search paths;
- native error translation;
- upcall lifetime management;
- callback thread-attachment policy;
- native-access diagnostics;
- safe pointer wrappers and ownership tags.

Bindings must be generated from repository-controlled machine-readable declarations or curated headers. Generated files are deterministic and reviewed. Do not scatter handwritten downcall descriptors across backend code.

### 18.6 Native C ABI

JTorch-owned native kernels expose a narrow C ABI:

```c
uint32_t jtorch_abi_version(void);
const char* jtorch_backend_build_id(void);
int32_t jtorch_last_error(char* buffer, size_t capacity);
int32_t jtorch_cpu_gemm_f32(const jt_tensor_view* a,
                            const jt_tensor_view* b,
                            jt_tensor_view* out,
                            const jt_gemm_options* options);
```

Rules:

- fixed-width integer types;
- explicit struct size/version fields;
- no exceptions across the boundary;
- no C++ standard-library types;
- no ownership ambiguity;
- every pointer carries documented lifetime and address-space requirements;
- sanitizers and ABI compatibility tests run on native code.

### 18.7 CUDA backend

Final CUDA support includes:

- driver/runtime discovery through FFM;
- device properties and capability selection;
- caching allocator and allocation snapshots;
- streams, events, synchronization, and stream-safe destruction;
- pinned host memory;
- cuBLAS/cuBLASLt;
- cuDNN;
- NCCL integration through distributed;
- JTorch custom CUDA kernels through a C ABI;
- graph capture where supported;
- AMP, FP8 where hardware permits, and quantized kernels;
- profiler correlation;
- deterministic-algorithm and known-nondeterminism reporting.

Bring-up order is transfer, elementwise, reductions, matmul, normalization, softmax, convolution, attention, optimizers, then broader operator coverage.

### 18.8 ROCm backend

ROCm mirrors CUDA’s semantic contracts using HIP, rocBLAS/hipBLASLt, MIOpen, RCCL, and compatible custom kernels. CUDA-specific assumptions may not leak into common accelerator abstractions.

### 18.9 macOS backend policy

The required macOS baseline is CPU execution on Apple Silicon and Intel. A Metal/MPS backend is a research-to-production track with its own capability ledger. macOS support must never mean “compiles but cannot execute tensors.”

### 18.10 Other accelerators

XPU, Vulkan, OpenCL, and custom devices implement the same backend conformance suite. Device discovery alone does not qualify a backend as supported.

## 19. Distributed runtime

### 19.1 Foundation

Provide:

- rendezvous and key-value `Store`;
- process groups;
- asynchronous `Work` handles;
- collectives: broadcast, all-reduce, reduce, all-gather, gather, scatter, reduce-scatter, all-to-all, barrier, send/recv;
- rank/world/group semantics;
- timeout, cancellation, and failure diagnostics;
- backend plugins such as NCCL, Gloo-like CPU transport, MPI, and oneCCL where available.

### 19.2 DDP

Implement:

- parameter/buffer synchronization;
- gradient buckets;
- overlap of communication and backward;
- unused-parameter handling;
- communication hooks;
- static-graph optimization;
- deterministic state-dict behavior;
- parity tests across multiple local processes.

### 19.3 Device mesh and distributed tensor

Define placements such as replicate, shard, and partial over an N-dimensional device mesh. Distributed operators propagate placements and insert collectives through explicit rules.

### 19.4 FSDP/sharded training

Final sharded training includes:

- parameter sharding and all-gather/reduce-scatter;
- mixed precision;
- CPU/offload policy;
- prefetching;
- original-parameter views/identities;
- sharded/full/local state dictionaries;
- optimizer-state sharding;
- nested modules and shared parameters;
- distributed checkpoint integration.

### 19.5 Parallelism libraries

Tensor, pipeline, sequence/context, and expert parallelism are built from DeviceMesh/DTensor and collectives. They are not separate ad hoc communication stacks.

### 19.6 Distributed checkpoint

Use a storage-writer/reader abstraction, planners, sharded metadata, reshard-on-load, asynchronous staging, checksums, and failure recovery. Java object serialization is forbidden.

---

## 20. Mixed precision, quantization, sparsity, and structured tensors

### 20.1 AMP

Provide device-specific autocast policies and `GradScaler` behavior:

- nested scopes;
- op allow/deny/promote policies;
- unscale before clipping;
- overflow detection;
- skipped steps;
- distributed scaler synchronization;
- compiler compatibility.

### 20.2 Quantization (`jtorch-ao`)

Support evolving AO workflows without freezing them into the tensor core:

- observers and fake quantization;
- PTQ and QAT;
- dynamic/static quantization;
- per-tensor/per-channel/groupwise schemes;
- int8 and weight-only int4;
- float8 scaling strategies;
- packed weights and backend-specific kernels;
- graph-mode transformations;
- model-level accuracy tests.

### 20.3 Sparse and nested

Each layout needs explicit operator coverage, autograd coverage, conversion rules, invariants, and compiler behavior. Dense fallback must be opt-in for operations where it can explode memory.

---

## 21. Profiler and observability

Provide:

- CPU scopes and operator events;
- accelerator activities;
- shape/dtype/module metadata;
- memory allocation/free events;
- distributed collective events;
- trace export;
- configurable logging;
- debug dispatcher table and kernel-selection diagnostics;
- anomaly and NaN/Inf diagnostics;
- benchmark integration.

Profiling must have near-zero disabled overhead and must not alter synchronization semantics.

---

## 22. Demo application suite

The demos are release-gated applications, not screenshots or isolated examples. They use a shared zero-dependency Swing/AWT shell with Java2D rendering and JTorch public APIs. JavaFX is not required. A future optional UI adapter may exist, but the official demos must launch with the packaged JDK runtime alone.

### 22.1 Common application platform

`demos/jtorch-demo-common` provides:

- application lifecycle and crash reporting;
- cross-platform preferences using `java.util.prefs` with an export/import path;
- native menu integration where supported;
- custom accessible Swing components and keyboard navigation;
- high-DPI scaling and dark/light themes;
- task progress, cancellation, and structured error surfaces;
- backend/device chooser;
- model manifest, license display, download, resume, checksum, and cache management;
- local-only/privacy indicators;
- log viewer and diagnostic bundle export;
- deterministic `--smoke-test --headless` protocol;
- `--model-dir`, `--device`, `--offline`, and `--output-json` CLI options;
- no privileged use of JTorch internals.

Every demo command supports:

```text
--help
--version
--diagnostics
--smoke-test
--headless
--offline
--model-dir <path>
--device cpu|cuda:N|rocm:N|auto
--output-json <path>
```

`--smoke-test` must execute a tiny real JTorch graph or fixture model, validate the result, emit machine-readable JSON, and exit non-zero on semantic failure. It must not merely instantiate the main class.

### 22.2 Model acquisition policy

Large models are not committed to Git. Each model manifest records:

- stable logical model ID and version;
- architecture adapter;
- original source and revision;
- redistribution and use license;
- expected files, sizes, and SHA-256 hashes;
- tokenizer/preprocessing versions;
- minimum JTorch operator-set and backend capabilities;
- optional quantization variants;
- expected reference outputs for smoke inputs.

Downloads occur only after explicit user consent. Offline use is supported after acquisition. CI uses tiny repository-owned fixtures or generated deterministic weights, never multi-gigabyte public models on every pull request.

### 22.3 TensorLab — training and autograd workbench

**Purpose:** teach, inspect, and validate real model training.

Capabilities:

- train an MLP and small CNN on MNIST-compatible data;
- draw a digit and classify it using the just-trained model;
- inspect loss, accuracy, learning rate, gradients, parameter histograms, and memory;
- pause, resume, cancel, checkpoint, reload, and reproduce from seed;
- switch CPU reference/optimized/CUDA backends where supported;
- compare eager and compiled training results;
- export a safe JTorch checkpoint and Safetensors state;
- display autograd graph summaries and anomaly diagnostics.

Framework coverage:

- data loading and batching;
- random generators;
- convolution, linear, activations, losses;
- reverse-mode AD and optimizer state;
- checkpointing;
- profiler;
- compiler/AOT autograd in later milestones.

Acceptance:

- reaches a documented validation-accuracy threshold on the pinned dataset;
- deterministic smoke mode trains at least two steps and verifies loss decreases for a controlled fixture;
- interrupted training resumes with matching optimizer and RNG state;
- packages and launches on all three desktop operating systems.

### 22.4 Vision Inspector — local image classification and explainability

**Purpose:** inspect and batch-classify local images without cloud upload.

Capabilities:

- drag/drop or open PNG, JPEG, BMP, and supported ImageIO formats;
- run ResNet/ConvNeXt/ViT-class imported models;
- top-k labels and calibrated confidence display;
- batch-folder classification with CSV/JSON export;
- saliency or gradient-based attribution overlay;
- preprocessing visualization and model metadata;
- CPU/CUDA/ROCm comparison and latency measurements;
- optional camera capture only when a platform adapter is available.

Framework coverage:

- image tensor transforms;
- convolution and attention;
- model import/state dictionaries;
- autograd inference explanations;
- batching and profiler.

Acceptance:

- matches reference top-k outputs for the pinned fixture set;
- saliency path produces finite gradients and aligned output dimensions;
- corrupt images fail safely without crashing the batch;
- headless mode classifies a generated fixture image.

### 22.5 Local Semantic Search — private code, log, and note retrieval

**Purpose:** index a local folder and find relevant content semantically without sending files to a server.

Default supported file types:

- `.txt`, `.md`, `.rst`;
- `.java`, `.kt`, `.scala`, `.py`, `.rs`, `.c`, `.cpp`, `.h`;
- `.json`, `.yaml`, `.yml`, `.toml`, `.xml`;
- `.log` and configurable plain-text formats.

Capabilities:

- recursive folder indexing with ignore patterns and size limits;
- chunking, embedding, normalized vector storage, and incremental re-indexing;
- hybrid lexical plus embedding ranking;
- result previews with source path and line ranges;
- query history stored locally;
- index encryption is an optional later feature, never falsely implied;
- explicit removal and cache purge;
- no network access after model download.

Framework coverage:

- transformer encoder inference;
- tokenizer;
- batched embeddings;
- matrix/vector similarity;
- safe persistent tensor/index format;
- data pipeline and concurrency.

Acceptance:

- benchmark corpus retrieval metrics meet a documented baseline;
- file changes update only affected chunks;
- symlink cycles, binary files, huge files, and permission errors are handled safely;
- smoke mode indexes generated documents and retrieves the known relevant one.

### 22.6 GC Doctor — JVM garbage-collection analysis

**Purpose:** analyze real HotSpot unified GC logs and identify pause, allocation, promotion, humongous-allocation, evacuation, and memory-pressure patterns.

Capabilities:

- parse supported G1, ZGC, Shenandoah, Parallel, Serial, and Epsilon unified-log events with declared coverage;
- show pause percentiles, throughput, allocation rate, live-set trend, promotion pressure, heap occupancy, and anomaly windows;
- combine deterministic domain rules with an optional JTorch anomaly/classification model;
- highlight exact source log ranges supporting each conclusion;
- compare two runs and export an HTML/JSON report;
- avoid presenting model output as certainty;
- operate fully offline.

Framework coverage:

- sequence/tabular model inference;
- data normalization;
- batched window scoring;
- explainability/feature importance;
- serialization and model versioning.

Acceptance:

- parser golden tests cover supported log dialects and malformed/truncated files;
- every recommendation is traceable to metrics or model features;
- fixture logs produce stable known findings;
- the application remains useful in deterministic metrics-only mode when the ML model is unavailable.

### 22.7 JTorch Transcribe — offline speech transcription

**Purpose:** transcribe local recordings into timestamped text without cloud services.

Capabilities:

- WAV/AIFF/PCM input through standard Java sound APIs;
- resampling, log-mel spectrograms, Whisper-class encoder-decoder inference;
- language selection/detection where model support exists;
- segment timestamps;
- TXT, SRT, VTT, and JSON export;
- cancellation, progress, and partial result streaming;
- optional microphone capture behind explicit permission and platform support;
- quantized model variants for CPU systems.

Framework coverage:

- FFT/signal preprocessing;
- transformer encoder-decoder;
- KV cache and beam/greedy decoding;
- quantized inference;
- streaming and profiler.

Acceptance:

- word/segment output matches a pinned reference within declared text-normalization rules;
- timestamp ordering and bounds are valid;
- unsupported codecs report a remediation path rather than silently failing;
- smoke mode transcribes a tiny generated or licensed audio fixture.

### 22.8 JTorch Local Chat — private LLM inference

**Purpose:** run a local language model with streaming output and transparent resource use.

Capabilities:

- load supported Safetensors and GGUF model families through architecture adapters;
- tokenizer/chat-template handling;
- streaming generation;
- configurable temperature, top-k, top-p, min-p, repetition penalty, seed, and stop sequences;
- KV-cache inspection and context management;
- model/device/memory diagnostics;
- conversation save/export without hidden cloud synchronization;
- optional retrieval from Local Semantic Search through a public extension API;
- quantized CPU and accelerated GPU execution.

Framework coverage:

- embeddings, RMSNorm/LayerNorm, RoPE, GQA/MQA, attention, SwiGLU/MoE where declared;
- quantized matmul;
- KV cache;
- sampling RNG;
- tokenizer and model formats.

Acceptance:

- fixed-seed logits and generated tokens match reference fixtures within declared tolerance;
- stop-sequence and context-limit behavior is tested;
- cancellation frees temporary/KV-cache resources;
- smoke mode runs a tiny transformer fixture, not canned text.

### 22.9 JTorch Image Studio — local diffusion generation

**Purpose:** demonstrate production-relevant diffusion inference and image processing.

Capabilities:

- text-to-image with supported compact and full diffusion pipelines;
- negative prompts, seed, scheduler, steps, guidance, dimensions, and batch count;
- progress previews where the model allows;
- image-to-image and inpainting after the base pipeline is stable;
- PNG export with optional reproducibility metadata;
- CPU operation for small fixtures and GPU operation for real models;
- explicit memory estimation and out-of-memory remediation.

Framework coverage:

- text encoder;
- UNet/transformer denoiser;
- VAE encode/decode;
- convolution, attention, schedulers;
- mixed precision and memory planning.

Acceptance:

- latent and decoded fixture outputs match reference checkpoints within tolerance;
- deterministic seed reproducibility is tested per backend;
- smoke mode executes a tiny diffusion network and writes a valid image;
- real-model release acceptance includes a documented licensed model and visual regression set.

### 22.10 Demo UI and UX quality gates

All demos must provide:

- responsive layout at 100%, 150%, and 200% scale;
- keyboard-only navigation;
- accessible names/descriptions and logical focus order;
- cancelable long operations;
- no blocking work on the Swing event dispatch thread;
- progress that reflects real work units where measurable;
- actionable errors with diagnostic detail available but not dumped into primary UI;
- first-run model/license explanation;
- safe default output paths;
- persistence migration tests;
- clean shutdown with no lingering non-daemon workers;
- platform-native application icons and metadata;
- no tracking or telemetry by default.

### 22.11 Demo build contract

Every demo defines Gradle tasks:

```text
:<demo>:compileJava
:<demo>:test
:<demo>:smokeTest
:<demo>:runtimeImage
:<demo>:packageAppImage
:<demo>:distZip
```

The root defines:

```text
demoCheck          test + smokeTest for every declared demo
demoPackage        create platform app-image for every declared demo
demoDist           archive every platform app-image
demoList           print declared demos and required capabilities
```

A demo may be added to `settings.gradle.kts` only when its first real vertical slice and smoke test exist. Once declared, it is mandatory in cross-platform CI.

---

## 23. GitHub Actions and release engineering

GitHub workflows must build JTorch and demos together. A green core build with broken demos is a failed build.

### 23.1 Supported runner matrix

Required stable matrix:

```yaml
include:
  - os: ubuntu-24.04
    platform: linux
    arch: x64
  - os: windows-2025
    platform: windows
    arch: x64
  - os: macos-15
    platform: macos
    arch: arm64
  - os: macos-15-intel
    platform: macos
    arch: x64
```

Additional nightly/experimental entries may include `ubuntu-24.04-arm` and `windows-11-arm`, but public-preview runners must not block stable releases until explicitly promoted.

Never use `*-latest` for release artifacts. Pin explicit runner labels and periodically update them through a reviewed infrastructure change.

### 23.2 Reusable build setup action

`.github/actions/setup-jtorch-build/action.yml` must:

- install Temurin or another project-approved JDK 25 distribution using `actions/setup-java`;
- validate `java --version` and architecture;
- enable Gradle caching through supported action inputs;
- validate the Gradle wrapper checksum;
- print runner image and tool versions;
- configure Python only when the caller requests oracle support;
- never download demo production models implicitly.

### 23.3 Pull-request CI

`.github/workflows/ci.yml` is the mandatory gate:

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]

concurrency:
  group: ci-${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

permissions:
  contents: read

jobs:
  build-test-demos:
    name: ${{ matrix.platform }}-${{ matrix.arch }}
    strategy:
      fail-fast: false
      matrix:
        include:
          - os: ubuntu-24.04
            platform: linux
            arch: x64
          - os: windows-2025
            platform: windows
            arch: x64
          - os: macos-15
            platform: macos
            arch: arm64
          - os: macos-15-intel
            platform: macos
            arch: x64
    runs-on: ${{ matrix.os }}
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: ./.github/actions/setup-jtorch-build
      - name: Verify generated sources
        shell: bash
        run: ./gradlew --no-daemon clean generateAll verifyGeneratedClean
      - name: Build JTorch and demos
        shell: bash
        run: ./gradlew --no-daemon check demoCheck
      - name: Package demo app images
        shell: bash
        run: ./gradlew --no-daemon demoPackage
      - name: Archive test reports
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: reports-${{ matrix.platform }}-${{ matrix.arch }}
          path: |
            **/build/reports/**
            **/build/test-results/**
          if-no-files-found: ignore
          retention-days: 14
```

Because Windows runners do not provide a POSIX shell for every context by default, the repository must either include a verified Bash environment path or use platform-neutral Gradle invocation steps. The implemented workflow may split Unix and Windows command steps; the semantic task set must remain identical.

Required CI checks:

- compile all production modules;
- unit/schema/meta/autograd tests;
- fast pinned-oracle parity subset;
- root `demoCheck`;
- `jpackage --type app-image` for every declared demo;
- JPMS module-resolution test;
- generated-source cleanliness;
- dependency verification;
- no undeclared runtime dependency in zero-dependency modules;
- native-access launch smoke for FFM modules;
- packaging path and spaces-in-path test.

### 23.4 Dedicated demo workflow

`.github/workflows/demos.yml` runs on PR changes to demo, ecosystem, serialization, model-manifest, packaging, or public API paths and on manual dispatch. It must:

- execute every demo headless smoke test on all stable platforms;
- build platform app images;
- launch each packaged image in smoke/headless mode rather than only running classes from Gradle;
- upload logs, JSON outputs, generated images/audio/text, and package archives;
- use tiny fixture models only;
- enforce that no demo attempts undeclared network access under `--offline`.

### 23.5 Oracle workflow

`.github/workflows/oracle.yml` runs on Linux for speed and reproducibility:

- installs the exact Python and PyTorch versions from `compat/baseline.toml`;
- verifies package hashes/lock data;
- regenerates selected fixtures;
- fails if committed fixtures or compatibility reports differ;
- runs full differential suites nightly and a bounded suite on PRs;
- never uses unpinned `latest` packages;
- publishes a compatibility delta artifact.

### 23.6 Native workflow

`.github/workflows/native.yml` builds JTorch-owned native CPU libraries and verifies FFM bindings on all stable target combinations. Required checks:

- C ABI version test;
- exported symbol allowlist;
- AddressSanitizer/UndefinedBehaviorSanitizer on supported Linux/macOS jobs;
- Windows runtime checks with the supported compiler toolchain;
- generated FFM descriptor consistency;
- load/unload/reload and error-path tests;
- architecture mismatch rejection;
- native library packaged inside demo runtime images only when selected.

No JNI artifact may appear in build outputs.

### 23.7 GPU workflow

`.github/workflows/gpu.yml` uses explicitly labeled self-hosted or approved larger runners:

```text
self-hosted, linux, x64, cuda
self-hosted, linux, x64, rocm
```

It runs on manual dispatch, protected-branch schedules, and backend changes. It must not run untrusted fork pull-request code on privileged self-hosted runners.

Checks include:

- device discovery;
- allocator/stream/event tests;
- parity subset;
- model-level training/inference;
- demo smoke tests on supported GPU demos;
- leak/fragmentation loop;
- multi-GPU collectives when hardware permits;
- driver/library version report.

### 23.8 Nightly workflow

`.github/workflows/nightly.yml` runs:

- full operator differential suite;
- randomized/fuzz tests;
- long memory/concurrency stress;
- compiler dynamic-shape suites;
- distributed fault injection;
- all demo real-model acceptance tests where licenses and runner resources permit;
- optional ARM runner matrix;
- benchmark baselines without automatically treating noisy hosted-runner results as release blockers;
- security scans and hostile-file corpus;
- compatibility report generation.

Nightly failures open or update a deduplicated issue with artifacts and the first failing baseline commit.

### 23.9 Code scanning and supply-chain workflow

`.github/workflows/codeql.yml` scans Java/Kotlin build logic and native code where supported. Additional requirements:

- Gradle dependency verification enabled;
- action versions pinned to reviewed major or commit policy;
- minimal workflow permissions;
- no secret exposure to fork PRs;
- model download manifests signed or checksum-pinned;
- release checksums and provenance artifacts;
- no arbitrary model deserialization.

### 23.10 Release workflow

`.github/workflows/release.yml` triggers on signed version tags and manual release candidates. It:

1. verifies the tag, clean generated sources, compatibility ledger, changelog, and release gates;
2. builds and tests all JTorch modules and demos on the stable matrix;
3. builds sources/Javadocs and Maven artifacts;
4. creates `jlink` runtime images and `jpackage --type app-image` packages for every demo;
5. optionally creates native installers only when tooling/signing is configured;
6. archives app images as ZIP on Windows and macOS and TAR.GZ on Linux;
7. signs/notarizes Windows/macOS artifacts when secrets and identities are configured;
8. emits SHA-256 checksums, SBOM/provenance, module graph, and model manifest bundle;
9. uploads artifacts to the GitHub release;
10. publishes Maven artifacts only after all platform jobs succeed.

Required artifact names:

```text
jtorch-<version>-modules.zip
jtorch-<version>-sources.zip
jtorch-<version>-javadocs.zip
<demo>-<version>-windows-x64.zip
<demo>-<version>-linux-x64.tar.gz
<demo>-<version>-macos-arm64.zip
<demo>-<version>-macos-x64.zip
SHA256SUMS
compatibility-report-<version>.zip
```

Release artifacts must not bundle large third-party models unless redistribution rights are explicit. The demo model manager downloads model assets separately.

### 23.11 `jlink` and `jpackage` rules

- Every demo is modular and produces a minimized JDK 25 runtime image with `jlink`.
- Use `jpackage --type app-image` as the universal CI packaging baseline.
- Windows MSI/EXE, macOS DMG/PKG, and Linux DEB/RPM are release enhancements and require platform tooling/signing.
- Native launcher JVM options include exact module native-access flags, memory defaults, crash-log path, and UTF-8 settings.
- Packaged smoke tests execute the generated launcher, not `java -cp`.
- Runtime images contain license/notice files and model-manifest metadata.
- Packaging must work from paths containing spaces and non-ASCII characters.

### 23.12 Required root Gradle task graph

```text
check
├── coreCheck
├── schemaCheck
├── parityFast
├── nativeFfmCheck
└── demoCheck
    ├── tensorlab:smokeTest
    ├── vision-inspector:smokeTest
    ├── semantic-search:smokeTest
    ├── gc-doctor:smokeTest
    ├── transcribe:smokeTest
    ├── local-chat:smokeTest
    └── image-studio:smokeTest

releaseCheck
├── check
├── parityFull
├── modelAcceptance
├── demoPackage
├── apiCompatibility
├── securityCheck
└── compatibilityReport
```

A release cannot exclude a broken declared demo to make the pipeline green. Temporarily unsupported capabilities must remain visible inside the demo with typed capability diagnostics, while the app’s verified core use case remains functional.

---

## 24. Public Java API principles

### 24.1 Naming

Use idiomatic Java:

```java
Tensor x = Torch.randn(Shape.of(32, 768), TensorOptions.defaults()
    .dtype(DType.FLOAT32)
    .device(Device.cpu())
    .requiresGrad(true));
```

Avoid a lowercase public class named `torch`. Provide static imports and concise factories instead.

### 24.2 Options and overloads

Generated overloads should cover common defaults without combinatorial explosion. Use immutable options objects for long keyword-only tails. Preserve overload identity in the internal schema.

### 24.3 Errors

Typed exceptions include:

- operator/schema;
- overload;
- argument names;
- shapes, strides, dtypes, layouts, and devices;
- selected dispatch keys and backend;
- expected/actual condition;
- remediation when known.

Do not expose raw backend error codes without context.

### 24.4 Kotlin and Scala

Core is Java-first and binary-stable. Optional Kotlin/Scala adapters may add extension syntax but may not become required runtime dependencies.

### 24.5 API stability

- pre-1.0 APIs may evolve with deprecation notes;
- 1.0 stable namespaces require semantic versioning and deprecation windows;
- experimental, beta, and stable labels are machine-readable;
- generated signatures are compared across releases for binary/source compatibility.

---

## 25. Testing and numerical correctness

JTorch is a numerical runtime. Line coverage alone is meaningless.

### 25.1 Test layers

Every feature uses the relevant subset of:

1. unit tests;
2. schema tests;
3. differential PyTorch parity tests;
4. meta/shape tests;
5. aliasing/mutation tests;
6. gradient and grad-gradient tests;
7. forward-AD tests;
8. transform-composition tests;
9. cross-backend tests;
10. serialization/interoperability tests;
11. compiler eager-vs-compiled tests;
12. distributed multiprocess tests;
13. randomized/metamorphic tests;
14. fuzz tests;
15. stress/leak/concurrency tests;
16. model-level acceptance tests;
17. performance regression tests.

### 25.2 Pinned oracle

`compat/baseline.toml` records:

- PyTorch version and source commit;
- Python version;
- backend versions;
- platform;
- default dtype and determinism settings;
- extraction tool version.

CI never compares against an unpinned `latest`. Updating the baseline is a dedicated compatibility release with a generated delta report.

### 25.3 Golden transport

The oracle writes:

- canonical JSON metadata;
- Safetensors or raw bounded binary tensor payloads;
- seed and generator state;
- strides/storage offsets/alias groups;
- expected values and gradients;
- expected exception category;
- per-case tolerance policy.

Never depend on NumPy `.npy/.npz` parsing in core tests unless an optional test adapter is used.

### 25.4 Numerical comparison

Do not use one global tolerance per dtype.

Each operator/backend policy may consider:

- absolute and relative error;
- ULP distance;
- tensor magnitude;
- reduction length;
- conditioning;
- algorithmic nondeterminism;
- equal-NaN and signed-zero semantics;
- complex real/imaginary error;
- statistical confidence for random ops.

Exact comparison is required for integer, boolean, index, shape, alias, and metadata results unless semantics say otherwise.

### 25.5 Gradient testing

Differentiable ops require:

- oracle gradient comparison;
- finite-difference checks in float64/complex128 when appropriate;
- grad-grad checks for higher-order support;
- broadcast reduction cases;
- non-contiguous/view cases;
- in-place invalidation cases;
- undefined/zero gradient semantics;
- multiple-output and non-scalar output cases.

### 25.6 Properties and metamorphic tests

Use only mathematically and numerically valid properties. Do not assert exact associativity of floating-point addition or matrix multiplication.

Good examples:

- shape/view round-trip when legal;
- `x + 0` within exact/defined semantics;
- transpose involution;
- sum of one-hot probabilities;
- decomposition vs native implementation;
- eager vs compiled;
- CPU reference vs backend;
- functionalized vs original mutation behavior;
- serialization round-trip.

### 25.7 Aliasing tests

Every view/in-place/out operator tests:

- storage identity;
- storage offset;
- strides;
- version counters;
- mutation visibility;
- autograd legality;
- overlap detection;
- functionalization result.

### 25.8 Random tests

Test generator state, seed reproducibility, distribution moments/quantiles, invalid parameter behavior, device consistency, and transform randomness modes.

### 25.9 Compiler tests

For captured graphs, test:

- eager equivalence;
- gradients;
- dynamic shapes and guard failures;
- graph breaks;
- alias/mutation functionalization;
- serialization and reload;
- unsupported custom operators;
- recompilation/cache behavior;
- memory-planner safety.

### 25.10 Distributed tests

Use deterministic local multiprocess tests plus fault injection:

- rank failure;
- timeout;
- collective mismatch;
- partial checkpoint;
- reshard on different world size;
- uneven inputs;
- shared parameters;
- communication overlap.

### 25.11 Memory and concurrency tests

Track:

- native allocation balance;
- storage reference counts;
- allocator reuse;
- view lifetime;
- async stream/event lifetime;
- repeated model load/unload;
- deep autograd graphs;
- concurrent read/dispatch;
- scoped grad/autocast/device/dispatch state;
- classloader/plugin unload where supported.

### 25.12 Performance tests

Benchmarks are separate from correctness gates but regressions are visible and eventually release-blocking.

Measure:

- allocation latency;
- elementwise bandwidth;
- reductions;
- GEMM/GEMV/batched matmul;
- convolution;
- normalization/softmax;
- attention;
- optimizer step;
- training step;
- token latency/throughput;
- compile time and graph-cache hit rate;
- memory peak and fragmentation;
- collective bandwidth/latency.

Publish hardware, software versions, warmup, repetitions, variance, and confidence intervals. Never publish cherry-picked single runs.

---

## 26. Security and robustness

Treat model files, exported graphs, custom operators, and distributed peers as untrusted unless explicitly configured otherwise.

Requirements:

- no arbitrary Java or Python object deserialization;
- bounds and overflow checks before allocation;
- archive entry/size/count limits;
- path traversal and symlink protection;
- checksum/signature hooks;
- operator allowlists for deployment;
- custom native library opt-in;
- safe temporary files;
- denial-of-service limits for symbolic shapes and graph complexity;
- redaction of sensitive paths/data from diagnostics where configured;
- dependency and native-binary provenance reports;
- reproducible release artifacts where feasible.

Security fixes may bypass ordinary deprecation policy.

---

## 27. Build, style, and quality rules

- Gradle wrapper is mandatory.
- JDK 25 toolchains and `--release 25` are mandatory for all production Java source sets.
- Enable strict compiler warnings; suppress narrowly with rationale.
- No Lombok, Spring, runtime DI framework, or annotation processor in core.
- Avoid hidden global mutable state. Grad mode, inference mode, autocast, device, stream, and dispatch context use `ScopedValue` or explicit scoped handles with nesting tests.
- Public methods require Javadoc and examples for non-obvious semantics.
- Generated code is deterministic and checked by CI for clean regeneration.
- Every module declares its runtime dependencies; core dependency checks fail the build if policy is violated.
- No benchmark code in production hot paths.
- No API changes without compatibility diff and ADR when architectural.
- No binary artifacts committed except small, reviewed golden fixtures.

### Standard commands

Codex must establish and maintain these commands:

```bash
./gradlew clean check demoCheck
./gradlew demoPackage
./gradlew parityFast
./gradlew parityFull
./gradlew compatibilityReport
./gradlew benchmark
./gradlew regenerateSchemas verifyGeneratedClean
./gradlew releaseCheck
```

Backend-specific commands may add environment requirements but must skip clearly when unavailable rather than report false success.

---

## 28. Documentation and decision records

`AGENTS.md` is the normative technical and execution source of truth. Do not create parallel architecture or roadmap documents that restate it and drift.

Allowed documentation artifacts:

- `README.md` for installation, quick start, supported status, and links into generated reports;
- machine-readable status and compatibility data under `compat/`;
- generated human-readable compatibility and benchmark reports under build/release artifacts;
- focused user tutorials and API Javadocs;
- `docs/adr/` only for new irreversible or cross-cutting decisions not already resolved here;
- focused operator, backend, custom-op, numerical-testing, security, and release guides that explain procedures rather than redefining architecture.

An ADR records context, decision, alternatives, consequences, migration, owner, and status. Once accepted, any durable rule that affects the whole project must also be reconciled into this file. Agents may not settle architecture only in chat logs.

---

## 29. Agent organization and ownership

The lead Codex thread owns architecture, integration, public contracts, and final validation. Delegate bounded work to subagents in isolated worktrees.

### Agent A — Schema, code generation, and compatibility

Owns `schema/`, `tools/opgen/`, `tools/oracle/`, `compat/`, generated wrappers, registries, derivative tables, and reports.

### Agent B — Tensor metadata, FFM memory, views, and aliasing

Owns tensor metadata, storage handles, `MemorySegment` host storage, allocators, layouts, views, version counters, and leak diagnostics.

### Agent C — Dispatcher, meta/fake tensors, and custom operators

Owns dispatch keys, priority, registration, redispatch, provider loading, meta kernels, decompositions, and extension APIs.

### Agent D — Autograd and function transforms

Owns reverse/forward AD, saved tensors, hooks, custom functions, JVP/VJP, vmap, functionalization, checkpointing, and transform composition.

### Agent E — CPU kernels, RNG, and generated JVM kernels

Owns reference kernels, optimized Java kernels, Class-File-API generation, CPU scheduling, RNG, and CPU benchmarks. Coordinates optional vector kernels without exposing incubating APIs.

### Agent F — NN, optimizers, losses, and data

Owns modules, parameters/buffers, state dictionaries, layers, losses, optimizers, schedulers, datasets, samplers, collators, virtual-thread I/O orchestration, and process workers.

### Agent G — Serialization and interoperability

Owns native format, Safetensors, restricted PyTorch conversion, exported-program adapters, ONNX, GGUF, hostile-file tests, and model manifests.

### Agent H — Compiler and export

Owns capture, symbolic shapes, IRs, functionalization integration, decompositions, AOT autograd, memory planning, generated kernels, export packages, and compile cache.

### Agent I — FFM/native ABI and accelerators

Owns `jtorch-native-ffm`, generated bindings, C ABI, native CPU libraries, CUDA, ROCm, allocator/stream/event semantics, and native sanitizers. JNI is explicitly outside scope.

### Agent J — Distributed runtime

Owns stores, process groups, collectives, DDP, meshes/distributed tensors, sharding, parallelism, distributed checkpointing, timeouts, and fault injection.

### Agent K — AMP, quantization, sparsity, and structured layouts

Owns autocast, gradient scaling, quantization, sparse/nested tensors, calibration, pruning, and deployment transformations.

### Agent L — Demo application platform and packaging

Owns `jtorch-demo-common`, accessibility, themes, model manager, headless protocol, `jlink`, `jpackage`, platform launchers, and cross-demo UI testing. It does not bypass JTorch APIs to make demos work.

### Agent M — TensorLab and Vision Inspector

Owns training workbench, vision preprocessing, model adapters, attribution, fixture models, and acceptance datasets.

### Agent N — Semantic Search and GC Doctor

Owns local indexing, embedding retrieval, GC-log parser/metrics, ML scoring adapters, traceable findings, and relevant acceptance corpora.

### Agent O — Transcribe, Local Chat, and Image Studio

Owns audio preprocessing/decoding, transformer generation, diffusion pipelines, model manifests, streaming/cancellation, and real-model acceptance. Split into separate worktrees when multiple agents are available.

### Agent P — CI, QA, security, release, and benchmarks

Owns workflow infrastructure, Gradle convention checks, cross-platform matrices, differential harness review, fuzz/stress/security suites, API compatibility, benchmark publication, release artifacts, and provenance.

### Coordination rules

- No two agents edit the same generated output concurrently.
- Shared schema/public API changes land through the lead before dependent branches rebase.
- Every agent returns changed files, exact tests, compatibility impact, risks, and commit hash.
- The lead reviews diffs and runs integration tasks; subagent success is not merge evidence.
- Generated files are changed only by generators.
- Demo agents must coordinate missing operators through schemas rather than adding private shortcuts.
- Native and demo packaging changes require at least one non-author platform CI result.
- If a design flaw is found, update this file or an ADR before propagating divergent fixes.

---

## 30. Change and merge gates

A change is done only when:

- implementation is complete for the claimed compatibility status;
- unit and differential tests cover normal, edge, and invalid cases;
- alias/autograd/transform tests exist where relevant;
- docs and compatibility ledger are updated;
- generated outputs regenerate cleanly;
- JDK 25 core checks and JPMS resolution pass;
- no unexplained JVM runtime dependency or native-access expansion was added;
- relevant benchmarks ran or the absence is justified;
- no sanitizer, leak, race, or security regression is known;
- public API compatibility report is reviewed;
- every declared demo still passes `smokeTest` and packages on required targets;
- FFM/native changes pass ABI, lifetime, and platform checks.

Review priorities:

1. memory safety and lifetime;
2. numerical/semantic correctness;
3. aliasing and autograd correctness;
4. concurrency and async-device correctness;
5. security of files/native loading;
6. API stability;
7. performance.

Formatting-only success is not a substitute for runtime tests.

---

## 31. Milestones and exit criteria

Milestones are capability gates, not dates. Parallel agents reduce elapsed time only after shared contracts and tests stabilize.

### M0 — JDK 25 repository, schema, CI, and demo shell foundation

Deliverables:

- Gradle 9.1+ wrapper running on JDK 25;
- JPMS module skeleton and dependency rules;
- schema format and deterministic generator;
- pinned PyTorch oracle environment;
- compatibility ledger/report generator;
- Linux x64, Windows x64, macOS arm64, and macOS x64 CI;
- `jtorch-demo-common` with headless smoke protocol;
- TensorLab application shell with a real scalar/tensor fixture path;
- app-image packaging proof on every platform.

Exit:

- clean checkout passes `check demoCheck demoPackage` on all stable targets;
- generated output is reproducible;
- one operator schema generates Java API, dispatcher registration, docs, and test descriptors;
- packaged TensorLab launcher executes `--smoke-test --headless`.

### M1 — Dense tensor semantics and FFM host memory

Deliverables:

- immutable Shape/Strides;
- `MemorySegment` native CPU storage, mapped storage, and debug allocator;
- dense strided tensors, offsets, views, and contiguity;
- dtype/scalar foundations;
- creation, copy, indexing, view, elementwise, reductions, and matmul reference kernels.

Exit:

- core operator set passes forward, metadata, alias, invalid-input, mapped-file, and cross-stride parity;
- allocation/lifetime stress is clean;
- tensors larger than array limits are structurally supported;
- TensorLab renders real tensor values and runs deterministic operations.

### M2 — Dispatcher, meta/fake tensors, and reverse autograd

Deliverables:

- prioritized dispatch-key set;
- registration and redispatch;
- meta kernels;
- reverse-mode engine, saved tensors, hooks, version validation;
- Linear, Parameter, MSE/CrossEntropy foundations, SGD.

Exit:

- first vertical slice in Section 0 passes PyTorch parity;
- eager/meta behavior agrees;
- mutation-after-save regression is detected;
- TensorLab trains a tiny network and resumes a checkpoint.

### M3 — NN, optimizers, data, serialization, and TensorLab release

Deliverables:

- module tree, parameters/buffers, hooks, train/eval;
- common modules/losses;
- SGD/Adam/AdamW and scheduler foundation;
- map/iterable datasets, samplers, collation, prefetch;
- native safe checkpoint and Safetensors state;
- MNIST-class TensorLab workflow.

Exit:

- MLP/CNN reaches documented accuracy;
- interrupted training reproduces after resume;
- TensorLab meets UI, accessibility, packaging, and smoke criteria;
- no core third-party JVM runtime dependency.

### M4 — Broad dense operator compatibility and Vision Inspector

Deliverables:

- indexing, scatter/gather, linalg, convolution, pooling, normalization, FFT foundations, and image transforms;
- module breadth needed for ResNet/ViT;
- ONNX/Safetensors model adapters;
- attribution gradients;
- Vision Inspector.

Exit:

- representative dense operator families reach declared full-semantics status;
- ResNet-class and ViT-class fixture models match references;
- Vision Inspector batch and saliency workflows pass cross-platform acceptance.

### M5 — Advanced autograd and function transforms

Deliverables:

- forward AD;
- higher-order gradients;
- JVP/VJP/Jacobian/Hessian;
- vmap and transform composition;
- functionalization;
- custom autograd functions and checkpointing.

Exit:

- declared transform matrix matches the oracle;
- aliasing/mutation workloads functionalize correctly;
- Vision attribution and advanced TensorLab diagnostics use public transform APIs.

### M6 — Transformer inference and Local Semantic Search

Deliverables:

- tokenizers;
- embeddings, attention, RoPE, GQA/MQA, RMSNorm, SwiGLU;
- encoder model adapters;
- safe embedding-index serialization;
- Local Semantic Search.

Exit:

- encoder logits/embeddings match reference fixtures;
- retrieval benchmark meets documented quality;
- incremental indexing and offline privacy tests pass;
- packaged search app works on all platforms.

### M7 — Optimized CPU, Class-File kernels, and FFM native CPU

Deliverables:

- tiled pure-Java kernels;
- generated specialized elementwise/fusion kernels;
- optional Vector API module;
- versioned C ABI and FFM binding generator;
- optional BLAS/oneDNN backend;
- profiler and memory diagnostics.

Exit:

- all optimized paths pass reference parity;
- FFM ABI/lifetime/sanitizer checks pass;
- published performance data includes methodology;
- demos automatically select optimized CPU without semantic changes.

### M8 — LLM generation, quantization, and Local Chat

Deliverables:

- decoder-only architecture adapters;
- KV cache;
- streaming generation and sampling;
- GGUF and quantized tensor support for declared formats;
- int8/int4 inference kernels;
- Local Chat.

Exit:

- fixed-seed logits/tokens match fixtures;
- memory/cancellation loops are leak-free;
- real supported compact model is demoable on CPU;
- Local Chat packages and passes offline tests.

### M9 — CUDA backend and AMP

Deliverables:

- FFM CUDA runtime/driver, allocator, streams/events;
- cuBLAS/cuBLASLt, cuDNN, custom kernels;
- autocast and gradient scaling;
- representative training/inference paths;
- GPU demo integration.

Exit:

- common conformance suite passes;
- TensorLab trains and Vision/Chat run on CUDA;
- async lifetime and OOM diagnostics pass stress;
- performance baselines are published honestly.

### M10 — Audio stack and JTorch Transcribe

Deliverables:

- audio tensor transforms, FFT/mel processing, resampling;
- encoder-decoder transformer and decoding;
- quantized CPU path;
- JTorch Transcribe.

Exit:

- pinned audio outputs/timestamps match reference policy;
- SRT/VTT exports validate;
- real licensed model acceptance and cross-platform packaging pass.

### M11 — Compiler, symbolic shapes, export, and generated runtime packages

Deliverables:

- proxy/fake capture;
- `SymInt`/guards;
- functional IR, primitive IR, passes, fusion, memory planner;
- AOT autograd;
- exported program and reload;
- compile integration in TensorLab and model demos.

Exit:

- eager/compiled values and gradients agree;
- dynamic shape and graph-break diagnostics pass;
- exported packages reload without Python;
- demo compile mode gives transparent fallback and diagnostics.

### M12 — Distributed training and checkpointing

Deliverables:

- store/process groups and CPU/GPU collectives;
- DDP;
- device mesh/distributed tensor;
- FSDP-style sharding and distributed checkpoint;
- fault injection and resharding.

Exit:

- multi-process training matches single-process reference;
- timeout/rank failure/partial checkpoint tests pass;
- supported GPU collective backend is documented.

### M13 — Diffusion, mixed precision breadth, and Image Studio

Deliverables:

- schedulers;
- UNet/transformer denoisers;
- VAE;
- text encoder integration;
- memory-efficient attention and precision policies;
- Image Studio.

Exit:

- latent and image fixtures match tolerances;
- deterministic seed tests pass per backend;
- real-model GPU workflow generates documented visual fixtures;
- app handles OOM and cancellation safely.

### M14 — GC Doctor and domain application hardening

Deliverables:

- unified GC log parser coverage;
- deterministic metrics engine;
- JTorch anomaly/classification model path;
- evidence-linked findings and comparison reports;
- GC Doctor release package.

Exit:

- parser golden corpus passes;
- known incidents are detected with measured precision/recall where ML is used;
- metrics-only mode remains useful;
- reports are reproducible and traceable.

### M15 — ROCm and structured tensor breadth

Deliverables:

- ROCm backend;
- sparse COO/CSR/CSC/BSR/BSC;
- nested/jagged tensors;
- PTQ/QAT and broader low-precision workflows;
- layout/compiler/backend integration.

Exit:

- representative sparse, jagged, quantized, and ROCm workloads meet correctness and accuracy budgets;
- coverage is explicit per operator/layout/backend.

### M16 — PyTorch-class 1.0 release candidate

Deliverables:

- stable namespaces and extension APIs;
- broad measured operator/module/optimizer/transform compatibility;
- production CPU and CUDA support, declared ROCm support level;
- compile/export/distributed maturity;
- all seven demos release-ready;
- full cross-platform packages, docs, profiler, benchmarks, compatibility and security reports.

Exit:

- all Section 32 gates pass;
- independent numerical, security, native-memory, and API audits complete;
- no open P0/P1 correctness, corruption, or memory-safety issue;
- every release workflow artifact is reproducible or provenance-attested.

## 32. JTorch 1.0 definition of done

JTorch 1.0 may be called a **PyTorch-class JDK 25 framework** only when all are true:

1. Core Java artifacts require JDK 25 and have no third-party JVM runtime dependencies.
2. Stable release artifacts use no preview APIs; Vector API support remains isolated and optional.
3. FFM is the only JTorch-owned native bridge, and no JNI artifact or source exists.
4. `MemorySegment`/allocator lifetime tests show no known leaks, use-after-free, or asynchronous destruction defect.
5. Dense eager tensor semantics, views, mutation, dtype promotion, devices, and layouts have broad measured compatibility.
6. Reverse/forward AD, higher-order gradients, function transforms, functionalization, and custom autograd work on the declared stable operator set.
7. Modules, losses, state dictionaries, optimizers, schedulers, data loading, hooks, and checkpoint resume are production-usable.
8. CPU reference and optimized CPU paths pass the same conformance suite on Linux, Windows, macOS arm64, and macOS x64.
9. CUDA supports representative vision, transformer, audio, and diffusion training/inference; ROCm status is explicitly reported.
10. AMP and documented quantization workflows meet correctness/accuracy budgets.
11. Compiler/export supports dynamic shapes, functionalization, AOT autograd, stable artifacts, and Python-free reload.
12. Safetensors and ONNX adapters are production-usable; PyTorch-export/GGUF support is versioned and explicit.
13. DDP, sharded/distributed training, collectives, and distributed checkpointing pass multiprocess failure tests.
14. Sparse/nested/quantized coverage is explicit and useful.
15. Compatibility reports are generated from pinned differential tests, not hand-maintained percentages.
16. Security defaults prohibit arbitrary object deserialization and unverified native/model loading.
17. Profiler, diagnostics, extension APIs, Javadocs, tutorials, and release artifacts are available.
18. Binary/source compatibility and deprecation policies are active.
19. TensorLab, Vision Inspector, Local Semantic Search, GC Doctor, JTorch Transcribe, Local Chat, and Image Studio are real, packaged, demoable applications.
20. Every demo passes headless fixture smoke tests and platform packaging in CI; real-model acceptance is documented where applicable.
21. GitHub release workflows build JTorch and demos together for Windows, Linux, macOS arm64, and macOS x64.
22. Native application images use minimized JDK 25 runtimes and exact module native-access configuration.
23. No release claim says “drop-in” without naming the compatibility tier and linking its generated report.

Literal unmodified Python source compatibility is a separate Tier E claim. JTorch 1.0 does not claim it unless a Python facade passes an unmodified-program conformance suite.

## 33. Non-goals and traps

Do not:

- equate loading GGUF with replacing PyTorch;
- claim Java source compatibility means existing Python source runs unchanged;
- retain Java 25/JNI architecture “just in case” after choosing JDK 25;
- expose incubating Vector API classes publicly;
- use preview JDK features in release artifacts;
- make every tensor `AutoCloseable`;
- close an `Arena` while a tensor/view/queued kernel can still reference its segment;
- treat `MemorySegment` as proof of zero-copy or automatic performance;
- implement hundreds of wrappers before aliasing, dispatch, and autograd foundations;
- call `switch(device)` a dispatcher;
- let demos import internal packages or bypass schemas;
- call a demo working because its window opens;
- use canned outputs in smoke tests;
- download multi-gigabyte models during ordinary PR CI;
- execute arbitrary pickle/model code;
- silently fall back from GPU to CPU;
- promise native installers without signing/toolchain handling;
- use `*-latest` runner labels for release builds;
- run untrusted fork code on self-hosted GPU runners;
- use virtual threads for dense numerical tiling;
- use one global tolerance per dtype;
- test invalid exact floating-point identities;
- hand-maintain generated overloads and bindings;
- add third-party Java UI frameworks to official demos without a reviewed decision;
- claim completeness from line count, scaffolding, or smoke tests alone.

The shortest credible path is metadata-driven schemas/codegen, exact tensor and alias semantics, layered dispatch, FFM memory/native boundaries, differential testing, decompositions, and demos that force vertical integration.

## 34. Initial implementation backlog

After inspection, create small dependency-ordered issues. The initial set should normally be:

1. Pin Gradle 9.1+ and JDK 25 toolchain; enforce no preview APIs.
2. Add JPMS module descriptors and dependency-boundary tests.
3. Create explicit Linux/Windows/macOS CI matrix and reusable setup action.
4. Create root `demoCheck`/`demoPackage` tasks and packaged launcher smoke harness.
5. Implement `jtorch-demo-common` and TensorLab fixture application.
6. Create pinned oracle manifest and compatibility ledger.
7. Define operator schema and generate one operator end-to-end.
8. Implement immutable Shape/Strides with overflow tests.
9. Implement DType/Scalar/Device/Layout/DispatchKeySet value types.
10. Implement `MemorySegment` host allocator, `StorageHandle`, TensorImpl, and Tensor.
11. Implement view/reshape/transpose/contiguous and alias/version tests.
12. Implement dispatcher registration, CPU/meta redispatch, and provider loading.
13. Implement meta and CPU reference kernels for creation/add/mul/sum/matmul.
14. Implement reverse autograd and derivative metadata for the same operators.
15. Implement Linear/Parameter/Module/MSE/SGD vertical slice.
16. Implement safe native checkpoint and resume.
17. Make TensorLab train a tiny deterministic model.
18. Package TensorLab with `jlink`/`jpackage` and run packaged smoke tests on all stable targets.
19. Generate compatibility and test reports as CI artifacts.
20. Add FFM native-access diagnostic test and prove no JNI output exists.

Do not start broad CUDA, distributed, diffusion, or Python-facade work before the first training/demo slice is semantically correct and cross-platform.

## 35. Final report format for every Codex run

Return:

1. **Implemented:** concrete behavior, not intentions.
2. **Architecture decisions:** ADRs created/changed.
3. **Files changed:** grouped by subsystem.
4. **Tests run:** exact commands and pass/fail/skip counts.
5. **Compatibility impact:** ledger entries changed and oracle baseline.
6. **Performance:** benchmarks run and environment, or why not applicable.
7. **Risks/limitations:** correctness, platform, API, and security caveats.
8. **Demo impact:** apps built, smoke tests, packages, and fixture/real-model status.
9. **Cross-platform status:** Linux, Windows, macOS arm64/x64 evidence.
10. **Next dependency-ordered tasks:** small enough for separate agents.

Never say “production-ready,” “drop-in,” or “parity” without pointing to the corresponding compatibility report and release gate.

---

## 36. Primary reference map

Use primary official sources and pin exact versions in the repository. Starting references:

### PyTorch

- PyTorch repository: https://github.com/pytorch/pytorch
- Native operator definitions: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml
- ATen native operator guide: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md
- Autograd derivative metadata: https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml
- Decompositions: https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py
- Dispatcher/custom backend guide: https://docs.pytorch.org/tutorials/advanced/extend_dispatcher.html
- Custom operators/library: https://docs.pytorch.org/docs/stable/library.html
- Autograd: https://docs.pytorch.org/docs/stable/autograd.html
- Function transforms: https://docs.pytorch.org/docs/stable/func.html
- Compile: https://docs.pytorch.org/docs/stable/generated/torch.compile.html
- Export: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/api_reference.html
- Data loading: https://docs.pytorch.org/docs/stable/data.html
- Distributed: https://docs.pytorch.org/docs/stable/distributed.html
- FSDP and distributed tensors: https://docs.pytorch.org/docs/stable/fsdp.html
- AMP: https://docs.pytorch.org/docs/stable/notes/amp_examples.html
- Serialization: https://docs.pytorch.org/docs/stable/notes/serialization.html
- ONNX: https://docs.pytorch.org/docs/stable/onnx.html

### OpenJDK/JDK 25

- JDK 25 project and feature list: https://openjdk.org/projects/jdk/25/
- Foreign Function & Memory API finalization: https://openjdk.org/jeps/454
- Scoped Values finalization in JDK 25: https://openjdk.org/jeps/506
- Class-File API: https://openjdk.org/jeps/484
- Vector API tenth incubator: https://openjdk.org/jeps/508
- JNI restriction direction: https://openjdk.org/jeps/472
- JDK 25 API docs: https://docs.oracle.com/en/java/javase/25/docs/api/

### Build and CI

- Gradle Java compatibility: https://docs.gradle.org/current/userguide/compatibility.html
- Gradle toolchains: https://docs.gradle.org/current/userguide/toolchains.html
- GitHub Actions matrix strategies: https://docs.github.com/actions/writing-workflows/choosing-what-your-workflow-does/running-variations-of-jobs-in-a-workflow
- GitHub-hosted runner labels: https://docs.github.com/actions/reference/runners/github-hosted-runners
- Workflow syntax: https://docs.github.com/actions/reference/workflows-and-actions/workflow-syntax
- `actions/setup-java`: https://github.com/actions/setup-java
- `jlink`: https://docs.oracle.com/en/java/javase/25/docs/specs/man/jlink.html
- `jpackage`: https://docs.oracle.com/en/java/javase/25/docs/specs/man/jpackage.html

When prose and implementation disagree, create a focused differential test and record observed behavior for the pinned versions. Do not infer compatibility from documentation alone.
