# AGENTS.md — JTorch

JTorch is a planned JDK 25-native, Java-first tensor and deep-learning library with PyTorch-compatible semantics: tensors with exact view/alias behaviour, a dispatcher, reverse-mode autograd, `nn` modules and optimizers, running on a pure-Java CPU reference backend first, with optional FFM-based native and GPU backends later.

**There is no code yet** — only this file, the README and the design. Nothing about builds or tests below is true until milestone M0 lands; don't describe the project as working anywhere.

The full design (tensor/storage model, dispatcher, autograd, compiler, backends, demo apps, CI matrix, testing strategy, milestones M0–M16) is [`docs/DESIGN.md`](docs/DESIGN.md); § numbers below refer to it.

## Where to start

Work the initial backlog in order (§34); the first target is the vertical slice in §0.2:

```java
Tensor x = Torch.tensor(new float[] {1, 2, 3, 4}, Shape.of(2, 2)).requiresGrad(true);
Tensor w = Torch.randn(Shape.of(2, 1), TensorOptions.defaults().requiresGrad(true));
Tensor loss = x.matmul(w).square().mean();
loss.backward();
SGD.builder().parameters(List.of(new Parameter(w))).learningRate(1e-2).build().step();
```

1. **M0:** Gradle 9.1+ wrapper on JDK 25 (Kotlin DSL, multi-project), JPMS module skeleton (`jtorch-api`, `jtorch-core`, `jtorch-tensor`, `jtorch-dispatch`, `jtorch-autograd`, `jtorch-cpu-reference`, …), CI on Linux, Windows and macOS, one operator generated end to end from a schema (§6).
2. **M1:** immutable `Shape`/`Strides`, `MemorySegment` host storage, dense strided tensors with views, creation/elementwise/reduction/matmul reference kernels (§8).
3. **M2:** dispatcher with CPU and meta keys, reverse-mode autograd with saved-tensor version checks, `Linear`/`Parameter`/MSE/SGD — the §0.2 slice matching a pinned PyTorch oracle (§9, §11, §25.2).
4. **M3:** modules, Adam/AdamW, datasets/loaders, safe checkpoints and Safetensors, an MLP/CNN on MNIST in the TensorLab demo.

Don't start CUDA, distributed training, the compiler, diffusion or the Python facade before M2 is correct on all three OSes (§34).

## Commands (target, once M0 exists)

```bash
./gradlew clean check demoCheck      # gradlew.bat on Windows
./gradlew parityFast                 # differential tests against the pinned PyTorch oracle
./gradlew regenerateSchemas verifyGeneratedClean
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

## Future SLOPOS/PARAMA targets and edge-efficiency requirements

**FUTURE TARGET ONLY / NOT VERIFIED.** jtorch is an independently authored candidate for **developer/runtime library, not a default SLOPOS desktop application**. Preserve design-only Java 25 tensor/autograd library with pure-Java CPU reference backend, no implementation yet and the current project's authoritative engineering contract, milestones, license and cross-platform requirements. The project is *not* a currently shipped SLOPOS/PARAMA app by virtue of being listed. Any optional first-party distribution requires a source/asset/provenance audit, real packaging, signed/verifiable builds, supported platform test matrix and no eager runtime/SDK load in the 512 MB core SLOPOS installation.

- **Linux/SLOPOS target:** Offer library/JAR/CLI examples to SLOPOS developer catalog on supported Java runtimes; pure Java CPU backend and test vectors should remain usable without system GPU or remote services. It is *not* a boot service nor a standalone first-party icon until an actual approved GUI exists.
- **Pi Zero 2 W (4× Cortex-A53, 512 MB shared RAM) efficiency:** Prefer correct small-tensor and stride/view operations with predictable allocation, no compulsory ML framework/native GPU dependency, zero-copy views only when aliasing semantics correct, bounded tensor allocations and thread pools with explicit memory preflight. Make CPU reference work without accelerator and avoid GPU-context startup when idle. Large transformer/diffusion workloads are not promised on a 512 MB board.
- **Future native PARAMA/SURYA target:** Future PARAMA compatibility contingent on proven Java 25 runtime, thread/memory/files/runtime support; pure Java CPU backend is the first plausible target, FFM/GPU backends require ported native ABIs, no assumption of a Linux JDK running as-is.
- **Evidence and error handling:** Before implementation design and eventually run a fixed small-tensor/autograd conformance benchmark on Pi Zero 2 W with a verified JDK; record correctness vs reference, peak heap/native RSS, allocator costs, p95 step latency and graceful oversized shape rejection.
- **Methodology:** Pin actual OS/ABI/runtime/GC/driver, reference input fixtures and revision; measure cold/warm time-to-usable, process-tree PSS/RSS, whole-system `MemAvailable`, steady idle CPU, local GUI response p95 if applicable, compute throughput and correctness independently, timeout/cancellation, memory-pressure failure and restart. Cross-build/virtualization is NOT proof of real Pi hardware or future PARAMA runtime. Decline or gate unsupported heavy operations before OOM; allow long computations but never permit them to starve critical interactive tasks. Keep external I/O delays distinct from app-induced UI stalls.
