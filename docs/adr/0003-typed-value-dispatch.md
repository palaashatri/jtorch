# ADR 0003: Typed value algebra and schema-validated dispatch

- Status: accepted (M2)
- Date: 2026-11-01

## Context

The design forbids raw `Object[]` in runtime contracts and requires a composable dispatcher with priority-ordered
dispatch keys, meta execution, autograd wrapping and composite decompositions. The full operator-schema generator was
needed early to avoid hand-maintained duplicates.

## Decision

- `schema/ops/aten.json` is the canonical operator input. `tools/opgen` deterministically generates operator ID
  constants, typed `Op` handles, docs, the compatibility ledger, a coverage report and parity descriptors.
- Kernels exchange `Value` subtypes (`TensorValue`, `IntValue`, `IntListValue`, `ScalarValue`, `DTypeValue`,
  `DeviceValue`, strings, bools, optional `None`). There is no `Object[]` in the runtime contract.
- `Dispatcher.call` validates schema arity, derives the backend key from tensor devices and adds `AUTOGRAD` when grad
  mode is on, the operator has a native formula, and an input requires grad. Wrappers redispatch after removing their
  key.
- Composite operators (`linear`, `softmax`, `log_softmax`, `mse_loss`) register one `COMPOSITE` kernel and get their
  shape rules and gradients from primitive operators, matching CompositeImplicitAutograd behaviour.
- Hand-written kernels remain hand-written (as in ATen); the schema governs signatures, metadata, documentation and
  the ledger rather than emitting kernel bodies.

## Consequences

- Duplicate kernel registration and missing coverage fail loudly; a test asserts every declared operator has the
  required CPU/meta/autograd registrations.
- The generator is checked in and `verifyGeneratedClean` fails on drift.
- Public `Tensor` overloads are still hand-written. A future milestone can generate facade overloads from the same
  schema without changing the dispatcher contract.
