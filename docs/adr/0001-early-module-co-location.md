# ADR 0001: Co-locate tensor metadata, storage and dispatch in `jtorch-core`

- Status: accepted (M0–M3)
- Date: 2026-11-01

## Context

The design topology lists `jtorch-tensor` below `jtorch-dispatch`, with `jtorch-api` beneath both. In Java's module
system a public `Tensor` method such as `matmul` must call the dispatcher, while the dispatcher must know about
`TensorImpl`. Splitting those types across two JPMS modules creates a compile-time cycle; package cycles are not
allowed.

## Decision

For M0–M3, `modules/jtorch-core` physically contains the value types, FFM storage, `Tensor`/`TensorImpl` and the
dispatcher. Packages remain separated (`org.jtorch.core`, `.memory`, `.dispatch`, `.autograd`, `.value`, `.spi`) so
that the projects can be split once the facade is generated and no longer needs a hand-written back-reference.

`jtorch-api` is the public facade above core (`Torch`), the CPU backend registers through the SPI, and the autograd
engine lives in its own module. Dependency direction for the extracted packages is unchanged.

## Consequences

- There is exactly one dispatcher path from public APIs to kernels.
- The build avoids an artificial cycle and the JPMS descriptor validates at compile time in every module.
- Moving `Tensor` in a later milestone must keep the generated facade free of back-references, or introduce an
  explicit handle/impl split with a service table; this ADR must be revisited then.
