# AGENTS.md — JTorch

JTorch is a JDK 25-native, Java-first tensor and deep-learning library with PyTorch-compatible semantics: tensors with exact view/alias behaviour, a dispatcher, reverse-mode autograd, `nn` modules and optimizers, running on a pure-Java CPU reference backend first, with optional FFM-based native and GPU backends later.

**Status:** the M0–M2 reference build and nearly all of M3 exist in this repository and are tested against PyTorch 2.11.0 fixtures on the development host (macOS arm64, JDK 25, Gradle 9.8.2). Scope and limitations are recorded in [`compat/`](compat/) and [ADR 0005](docs/adr/0005-supported-scope-m0-m3.md). Do not describe M4–M16 capabilities (CUDA, distributed, compiler, broader convolution/normalization, float16/bfloat16, vmap/forward-AD, Python parity) as working; they are design targets, not deliverables yet.

The full design (tensor/storage model, dispatcher, autograd, compiler, backends, demo apps, CI matrix, testing strategy, milestones M0–M16) is [`docs/DESIGN.md`](docs/DESIGN.md); § numbers below refer to it.

## Where to start

M0–M2 are complete and M3 is implemented except for prefetch and the UI/accessibility gate. Read the design, then work the remaining backlog in dependency order (§34). The completed first vertical slice is in §0.2 and runs from [`examples/vertical-slice`](examples/vertical-slice):

```java
Tensor x = Torch.tensor(new float[] {1, 2, 3, 4}, Shape.of(2, 2)).requiresGrad(true);
Tensor w = Torch.randn(Shape.of(2, 1), TensorOptions.defaults().requiresGrad(true));
Tensor loss = x.matmul(w).square().mean();
loss.backward();
SGD.builder().parameters(List.of(new Parameter(w))).learningRate(1e-2).build().step();
```

Completed in this build: M0 (Gradle wrapper, JPMS skeleton, schema generator, pinned oracle, CI workflow matrix, demo shell/packaging), M1 (shape/strides, FFM storage, strided views, reference kernels), M2 (dispatcher, meta, reverse autograd, Linear/MSE/SGD vertical slice) and most of M3 (modules, Adam/AdamW, tensor datasets and batching, checkpoints/Safetensors, TensorLab with checkpoint resume and packaged launcher). M3's CNN path and real-MNIST acceptance are now implemented (`Conv2d`, `MaxPool2d`, TensorLab `--mnist-dir`, 0.95 held-out threshold); prefetch and the UI/accessibility gate remain open.

Remaining design milestones: M4 broad dense operators and Vision Inspector, M5 transforms, M6 transformer inference/search, M7 optimized CPU, M8 LLM chat, M9 CUDA/AMP, M10 audio/Transcribe, M11 compiler/export, M12 distributed, M13 diffusion/Image Studio, M14 GC Doctor, M15 ROCm, M16 release candidate. Do not start GPU, distributed, compiler, diffusion or Python-facade work before the preceding capability gates are met (§34).

Known limitations that must stay documented: only float32/float64/int32/int64/bool, only CPU and meta execution, no in-place autograd on tensors requiring grad, no higher-order gradients, `prod` is forward-only, cross-entropy supports 2-D logits, convolution/pooling support groups=1/NCHW/ceil_mode=false only, optimizer state is not serialized, and TensorLab's default synthetic check uses a labelled synthetic set (real MNIST is available through `--mnist-dir`).

## Commands

```bash
./gradlew clean check demoCheck      # gradlew.bat on Windows
./gradlew parityFast                 # differential tests against the pinned PyTorch oracle
./gradlew demoPackage                # jpackage app image + packaged headless smoke test
./gradlew regenerateSchemas verifyGeneratedClean
./gradlew :examples:vertical-slice:run
./gradlew :demos:tensorlab:mnistCheck -PmnistDir=/path/to/mnist-idx
```

The oracle generators are optional Python tools and are never needed for a normal build:

```bash
python3 tools/oracle/generate.py
python3 tools/oracle/optimizers.py
python3 tools/oracle/safetensors_fixture.py
```

When you add or change a task, update this section in the same PR so it lists only commands that work.

## Settled decisions

- **JDK 25** class files and runtime; no preview features in release artifacts. The incubating Vector API lives only in `jtorch-cpu-vector` and never in public signatures (§3.1).
- **FFM only** for native interop (`MemorySegment`, `Arena`, `Linker`). No JNI anywhere in JTorch-owned code (§3.2).
- **Zero third-party runtime dependencies** in the core modules; PyTorch is used only as a test oracle, never at runtime (§3.4).
- The Java CPU reference backend defines semantics; optimized kernels must be switchable back to it (§3.5).
- Generated code (operator overloads, dispatcher registration, bindings) comes from the schema, never hand-maintained (§6).

## Rules

1. **No false completeness** (§3.6, §33): an operator counts only when forward values, shapes, strides, dtypes, aliasing and gradients match the oracle on CPU and meta. No stub wrappers, no silent CPU fallbacks or dtype coercions, no completion claims from line counts or scaffolding.
2. Public APIs never bypass the dispatcher; `switch(device)` is not a dispatcher.
3. Never tie an `Arena` to a scope a tensor or view can outlive; tensors are not generally `AutoCloseable` (§3.3).
4. Don't use virtual threads for numerical work, and don't use one global tolerance per dtype (§25.4).
5. Never execute pickle or other model code when loading weights (§16, §26).
6. Demo apps consume the published modules exactly as an outside user would; a window opening is not a passing demo (§22).
7. Keep the compatibility ledger under `compat/` accurate; record decisions as ADRs in `docs/adr/`.

## Git workflow

- Branch and open a PR per backlog item unless the owner explicitly asks for a direct push to `main`.
- Before pushing: `./gradlew check` passes on JDK 25. Say in the PR which OSes you ran it on and what you couldn't run.
- Don't add other planning Markdown files; status belongs in `compat/` and GitHub issues.

## Independent library portability and Raspberry Pi Zero 2 W efficiency

**JTorch is a numerical runtime/library, not a SLOPOS or PARAMA desktop application candidate.** It is explicitly **excluded from first-party bundled apps and the curated SLOPOS/PARAMA Software catalog**, including optional developer application entries. No SLOPOS launcher, icon, UI theme integration, platform-specific port or packaged OS app is required by this roadmap. Existing planned JTorch demos remain JTorch's own work and do not imply OS bundle eligibility.

- **Keep JTorch independent:** Prioritize Java 25 pure-Java CPU tensor, autograd, correctness and cross-platform JVM support. Any Java runtime independently available under a Linux/SLOPOS desktop can use JTorch as a normal library without SLOPOS-specific adapters. A hypothetical PARAMA JVM is an unrelated platform prerequisite, **not a JTorch PARAMA deliverable or current compatibility claim**.
- **Pi Zero 2 W (four Cortex-A53 cores, 512 MB RAM):** Treat small-tensor correctness/performance as a desired device profile when a suitable JDK 25 works there; do not assume that it currently does. Check actual shape-size and allocator overhead before allocation; use bounded worker pools, small storage footprints, correct tensor/view aliases, lazy optional CPU-vector/FFM/GPU modules and CPU reference fallback. Do not advertise large-model training under 512 MB.
- **Acceptance:** On any actual supported resource-constrained ARM64 Java platform, run reference tensor and gradient fixtures; measure JVM heap, off-heap and whole-process PSS/peak RSS, GC overhead, CPU utilization, allocation counts, step latency and correctness under pressure. A Pi-specific PASS requires real board and JVM evidence. Handle oversized shapes predictably before host OOM. Benchmarks and model comparisons must not trade away PyTorch-compatible semantics.
- **Governance:** Keep the status recorded in `compat/` and ADR 0005 accurate, keep `docs/DESIGN.md` as the canonical milestone plan, preserve application/demo separation, and make no unsupported platform claims. These are efficiency objectives, not permission to advance milestones or claim a shipped framework.

