# ADR 0002: Arena + Cleaner lifetime for native host storage

- Status: accepted (M1), revisit for M7 pooling
- Date: 2026-11-01

## Context

The design requires `MemorySegment` host storage, deterministic release and no use-after-free when views or saved
tensors outlive a wrapper. A reference-counted `StorageHandle` is the long-term plan. Ordinary tensor expressions
must not become `AutoCloseable`.

## Decision

`NativeStorage` owns an `Arena.ofShared()` and registers a `Cleaner` action that closes the arena once the storage
object is unreachable. Views, autograd saved tensors and node inputs keep the storage reachable through ordinary
references, so the arena cannot be closed while any tensor can still address it. Release is tied to reachability,
not to Java wrapper identity.

Debug counters (`CpuAllocator.liveBytes/liveAllocations/totalAllocatedBytes`) expose allocation accounting and a test
forces collections to prove bytes return to baseline.

`MappedStorage` wraps a mapped `ByteBuffer`; the file length must equal the declared tensor bytes, and `flush()`
forces dirty pages.

## Consequences

- No public tensor is `AutoCloseable`; a `TensorScope` can be added later without changing semantics.
- Release is not refcount-deterministic: memory is reclaimed after GC rather than at the last `release()` call.
- The reference allocator allocates one arena per tensor. M7 needs slab pooling; this ADR will be superseded.
- Zero-element tensors allocate one byte to keep a valid segment while reporting zero logical bytes.
