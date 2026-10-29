# JTorch

<img src="assets/brand/app-icon.svg" alt="JTorch icon" width="96" align="right" />

[Icon, wordmark, and brand assets](assets/brand/README.md)


PyTorch-style tensors, autograd and neural networks in pure Java (JDK 25): exact PyTorch semantics for shapes, strides, views and gradients, a CPU reference backend with no native dependencies, and optional native/GPU backends through the Foreign Function & Memory API instead of JNI.

**Status: design only — there is no code yet.** Nothing can be built or run today.

- Design, architecture, testing strategy and milestones: [docs/DESIGN.md](docs/DESIGN.md)
- Contributor and coding-agent instructions: [AGENTS.md](AGENTS.md)

## Roadmap

The design runs to a PyTorch-class 1.0 with CUDA, distributed training and a compiler (milestones M0–M16). That is many person-years of work, so the useful near-term product is much smaller. In order:

1. **First training step in Java** (milestones M0–M2): create tensors, run `matmul`/`square`/`mean`, call `backward()`, take an SGD step — with values and gradients checked against real PyTorch in CI on Linux, Windows and macOS.
2. **Train a real small model** (M3): `nn.Module`-style layers, Adam, data loaders, safe checkpoints and Safetensors, an MNIST MLP/CNN reaching a documented accuracy, shown live in the TensorLab demo app.
3. **Load real weights** (M4): Safetensors/ONNX adapters and enough convolution, pooling and normalization to run a ResNet-class image classifier.
4. **Fast CPU path** (M7): Vector API and FFM native kernels behind the same dispatcher, always switchable back to the reference kernels.
5. **Transformer inference** (M6/M8): tokenizers, attention, quantized weights, and small local LLM and embedding models.

GPU, distributed training and the compiler come after these work.

## How this differs from existing JVM options

Deep Java Library (DJL) and the Bytedeco PyTorch presets call the native PyTorch/LibTorch library; Deeplearning4j has its own native engine and API. JTorch's aim is different: PyTorch semantics implemented on the JVM itself, with a pure-Java reference backend and FFM (not JNI) for native acceleration. The first milestones are worth doing even as a learning and testing exercise for JDK 25's FFM and Vector APIs.

## Licence

No licence has been chosen yet, so the code is not yet open source in the legal sense. Add a `LICENSE` file (the design expects one, plus `NOTICE`) before accepting outside contributions.

## Independent JTorch library and edge efficiency

**JTorch is not a candidate for bundled SLOPOS/PARAMA desktop applications or their curated first-party Software catalogs.** It remains an independently distributed Java numerical library with its own examples and planned demos; no new desktop icon, OS-specific app wrapper or PARAMA port is required. If a Linux/SLOPOS device supports Java 25, library usage follows ordinary Java deployment.

The Raspberry Pi Zero 2 W remains an **aspirational small-tensor efficiency reference**, conditional on a compatible JDK. Prioritize bounded allocation, correct views/gradients, CPU-first operation, low memory footprint and honest rejection of out-of-memory tensor shapes. Keep measured resource tests distinct from large-model promises and from unverified board/OS compatibility. See [AGENTS.md](AGENTS.md#independent-library-portability-and-raspberry-pi-zero-2-w-efficiency).

