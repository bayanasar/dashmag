# dash-contract

The contract of Dashmag's operations, written once for every implementation of them. For each
operation it says what the operation computes and what it matches (PyTorch, transformers, fla,
flash-attn), the shapes, layouts and dtypes it takes and returns, what its masks and states mean
and what each of its options does; and it holds the options themselves, as plain data with their
defaults. It depends on nothing.

`dash-ops` implements the operations on Burn and re-exports these types where it has always had
them; an implementation on another framework implements the same contract with the same types.

| module | operation |
|---|---|
| `gated_delta` | the gated delta rule of Gated DeltaNet, chunked for training and recurrent for decoding, over a batch or packed variable-length sequences |
| `attention` | attention with grouped key-value heads and the masks models train with, and decoding against a key-value cache |
| `logprob` | the log-probability of chosen tokens under a large vocabulary, with the entropy |
| `loss` | cross-entropy fused with the output layer over a large vocabulary, with label smoothing, z-loss, an ignore index and masked ids |
| `conv1d` | causal depthwise conv1d from given states and over packed sequences, and its decoding step |
| `lowrank` | low-rank products `base + scale · (x U) V`, as LoRA's term |
| `lora` | PEFT's default adapter name, and how an adapter read in PEFT's format is given to a model's layers |
| `sample` | sampling the next token from a decoding step's logits: greedy, or drawn at a temperature with top-k, top-p and masked ids from a counter-based generator, with the token's log-probabilities |
| `clip` | gradients clipped by their global norm, as `torch.nn.utils.clip_grad_norm_` clips them |

The conventions every operation follows (layout, masks, dtypes, options, and how a call outside
an operation's contract is refused) are written out in the crate documentation (`src/lib.rs`).

Part of [Dashmag](https://github.com/bayanasar/dashmag) — a machine-learning library in Rust
on Burn. See the repository for the full picture.
