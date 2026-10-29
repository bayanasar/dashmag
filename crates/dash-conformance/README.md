# dash-conformance

The operation-level tests of Dashmag's operations, written once so that every implementation of
[`dash-contract`](../dash-contract) runs the same ones: each test body reads the values recorded
from what an operation matches (PyTorch, transformers, fla), runs its cases through an
implementation's adapter, compares what comes back and asserts it under the bounds measured when
the cases were recorded, printing every comparison.

An implementation's tests implement the crate's `Adapter` trait for a device, one method per
operation: it receives the inputs as f32 values on the host with the dtype each is to be given
in, runs the operation and the backward pass the test asks for through the implementation's own
autodiff, and returns the results read back to the host as f32. The crate documentation
(`src/lib.rs`) says exactly what an adapter receives and returns, and how it skips a case its
implementation does not take. Each test body returns what ran, what it skipped and which dtypes the
device lacked, for the implementation's test to assert. `dash-ops` runs these tests on Flex and on every CubeCL device
(`crates/dash-ops/tests/common/adapter.rs`) and asserts that every call ran; its own tests keep
what is about it alone: which of its kernels ran, its kernels against its reference, routing by
what the device has, one autodiff graph across calls and inputs read through their strides.

Covered so far: causal depthwise conv1d (widths, bias, activation, initial and final states with
their gradients, packed sequences, the decoding step), the gated delta rule (chunked, packed,
initial states and their gradients, the step), the chosen-token log-probability (masked ids,
chunking, its gradient) and attention (every mask, grouped heads, queries that see no key, the
gradients, decoding).

The fixtures stay where `dash-ops` keeps them, `crates/dash-ops/tests/fixtures`, read from this
crate's place in the repository; the crate is not published.

Part of [Dashmag](https://github.com/bayanasar/dashmag) — a machine-learning library in Rust
on Burn. See the repository for the full picture.
