# ADR 0004: Native checkpoint and Safetensors formats

- Status: accepted (M3)
- Date: 2026-11-01

## Context

Loading model state must never execute code. PyTorch pickle is explicitly out of scope. The design requires a
versioned native format, checksummed payloads and a Safetensors adapter.

## Decision

Native checkpoints (`*.jtch`) are:

1. eight bytes `JTORCH01`;
2. little-endian `int64` header length;
3. a UTF-8 JSON manifest with `format`, `version`, string metadata and per-tensor dtype/shape/offset/length;
4. concatenated little-endian C-order tensor payloads;
5. a trailing SHA-256 over every preceding byte.

Loading rejects bad magic, unsupported versions, out-of-bounds offsets, duplicate names, shape/byte mismatches,
truncation and checksum failures before materialising tensors. Safetensors follows the public specification, validates
contiguous offsets and exact file length, and ignores (does not write) metadata blocks.

## Consequences

- No pickle, no Python and no code execution on the load path.
- Supported dtypes are limited to float32/float64/int32/int64/bool; other dtypes fail explicitly.
- Whole files are read into memory and capped at 2 GiB; streaming and sharded loading are future work.
- The digest makes corruption detectable but is not a signature; provenance must come from model-manifest policy.
