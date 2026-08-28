# Zig LLM From Scratch Journey

A curriculum for learning LLM inference and tiny-model training in Zig without PyTorch.

This is a learning journey, not a code-generation checklist. The student should implement early-stage core pieces manually, run small tests, explain the math, and repeat the patterns until they stick. Later stages may use more assistance, references, and higher-level tools such as ZML.

## How to Use This File

For each task:

- Read the objective.
- Implement the smallest possible version.
- Write at least one tiny test with hand-checkable numbers.
- Explain the shapes and math in your own words.
- Only then move on.

Suggested status markers:

```text
[ ] not started
[/] attempted / in progress
[x] passed tests and explained
[?] blocked / needs review
```

---

# Stage 0 — Setup, Rules, and Baseline

## 0.1 Initialize the repository and Zig project

**Task**

Start from the smallest useful repository. If there is no VCS yet, initialize colocated Jujutsu/Git first, then initialize Zig:

```bash
jj git init --colocate .
zig init
```

The initial human-facing files should be:

```text
README.md      # short project intro, links to JOURNEY.md
JOURNEY.md     # this curriculum copied into the repo root
```

`zig init` will add the standard Zig starting files, typically:

```text
build.zig
build.zig.zon
src/
  main.zig
  root.zig
```

**Hints**

- If `.jj/` or `.git/` already exists, do not reinitialize VCS.
- Keep the README short at first; the curriculum belongs in `JOURNEY.md`.
- Use the generated Zig project before adding custom modules.
- Start with CPU and `f32` only.

**Success criteria**

- `jj status` works if Jujutsu is available.
- `zig build test` runs.
- `README.md` links to `JOURNEY.md`.
- You can print a tiny starter message from `zig build run`.

## 0.1b Create the learning code skeleton

**Task**

After the generated Zig project is working, gradually add files as the journey needs them:

```text
src/
  tensor.zig
  math.zig
  random.zig
  tokenizer.zig
  bigram.zig
  tests.zig
```

Do not create empty abstractions too far ahead. Add each file when a task needs it.

**Success criteria**

- The project still builds after each new file.
- Each new file has at least one test or is used by a tested module.

## 0.2 Establish learning rules

**Task**

Write a short note in the project README or local journey file answering:

1. Why am I avoiding PyTorch initially?
2. What parts do I want to understand deeply?
3. What am I allowed to use as a reference?

**Success criteria**

- You can explain the difference between “no PyTorch” and “no libraries at all”.
- You accept that early code may be slow and ugly if it teaches the math.

---

# Stage 1 — Raw Zig Math Primitives

Goal: understand the mechanics beneath neural networks using raw slices and explicit loops.

## 1.1 Vector operations

**Task**

Implement:

```text
vectorAdd(a, b, out)
vectorMul(a, b, out)
scalarMul(a, scalar, out)
dot(a, b) -> f32
```

**Hints**

- Require equal lengths.
- Decide whether functions assert, return errors, or both.
- Test with length 0, 1, and several elements.

**Success criteria**

- Hand-checkable tests pass.
- You can explain contiguous memory layout for a `[]f32`.

## 1.2 Matrix indexing

**Task**

Represent a 2D matrix in row-major `[]f32` memory and implement:

```text
index(row, col, cols) -> usize
get(matrix, row, col, cols) -> f32
set(matrix, row, col, cols, value)
```

**Hints**

- Row-major index is usually `row * cols + col`.
- Test a 2x3 matrix.

**Success criteria**

- You can draw where each element lives in memory.
- Tests catch swapped row/column mistakes.

## 1.3 Matrix multiplication

**Task**

Implement:

```text
matmul(A: [M, K], B: [K, N]) -> C: [M, N]
```

using explicit loops.

**Hints**

- Write the shape comment before writing code.
- Start with tiny matrices:

```text
[1 2] x [3] = [11]
        [4]
```

- Then test 2x2 and rectangular cases.

**Success criteria**

- Tests pass for at least three shapes.
- You can explain why the inner dimension `K` must match.

## 1.4 Numerically stable softmax

**Task**

Implement:

```text
softmax(logits, probs)
```

**Hints**

- Subtract max logit before exponentiating.
- Output probabilities should sum to approximately 1.
- Test equal logits: `[0, 0] -> [0.5, 0.5]`.

**Success criteria**

- Handles large logits without overflow, e.g. `[1000, 1000]`.
- You can explain why subtracting max does not change probabilities.

## 1.5 Cross entropy loss

**Task**

Implement:

```text
crossEntropy(logits, target_index) -> f32
```

**Hints**

- Use stable log-softmax if possible.
- For a one-hot target, loss is `-log(prob[target])`.

**Success criteria**

- Lower loss for higher target logit.
- You can explain why language modeling uses next-token cross entropy.

## 1.6 Sampling

**Task**

Implement:

```text
argmax(probs) -> token_id
sampleCategorical(probs, rng) -> token_id
temperature(logits, temp)
```

Optional later:

```text
topK
topP
```

**Hints**

- Use deterministic seeds in tests.
- Start with argmax before randomness.

**Success criteria**

- Sampling never returns a token with zero probability.
- You can explain greedy vs stochastic generation.

---

# Stage 2 — Tokenization and Dataset Basics

Goal: turn text into token IDs and batches.

## 2.1 Character or byte tokenizer

**Task**

Implement one of:

```text
character tokenizer: unique chars from dataset
byte tokenizer: values 0..255
```

**Hints**

- Byte-level is simpler because vocab is fixed.
- Character-level is more readable for tiny demos.

**Success criteria**

- `decode(encode(text)) == text` for simple examples.
- You can explain vocab size.

## 2.2 Next-token dataset

**Task**

Given token sequence:

```text
x = tokens[i .. i+T]
y = tokens[i+1 .. i+T+1]
```

Create training examples.

**Hints**

- Start with batch size 1.
- Use deterministic offsets before random sampling.

**Success criteria**

- For text `hello`, you can show the input and target token pairs.
- You can explain why target is shifted by one.

---

# Stage 3 — Bigram Language Model

Goal: train the smallest possible language model end-to-end.

## 3.1 Bigram forward pass

**Task**

Implement a bigram table:

```text
weights: [vocab_size, vocab_size]
logits = weights[current_token]
```

**Hints**

- This is just row lookup.
- No context beyond current token.

**Success criteria**

- Given token `t`, returns the `t`-th row of the table.
- You can explain what each row means.

## 3.2 Bigram loss over a sequence

**Task**

Compute average cross entropy over all next-token predictions.

**Hints**

- For each position, use current token as input and next token as target.
- Average over positions.

**Success criteria**

- Loss computes on a tiny string.
- You can identify which predictions are wrong.

## 3.3 Manual gradient for bigram model

**Task**

Implement gradient update for softmax + cross entropy:

```text
dlogits = probs
dlogits[target] -= 1
```

Apply this to the selected row.

**Hints**

- Average gradients if using multiple positions.
- Start with SGD.

**Success criteria**

- Training loss decreases on a tiny dataset.
- The model can overfit a repetitive string like `abababab`.
- You can explain why `probs[target] - 1` appears.

## 3.4 Text generation from bigram model

**Task**

Generate text token-by-token:

```text
current -> logits -> softmax -> sample -> next
```

**Hints**

- First use argmax.
- Then use categorical sampling.

**Success criteria**

- Model generates valid decoded text.
- On an overfit simple dataset, output resembles the pattern.

---

# Stage 4 — MLP Language Model

Goal: learn embeddings, hidden layers, activations, and more manual gradients before attention.

## 4.1 Embedding lookup

**Task**

Implement token embedding:

```text
embedding_table: [vocab_size, d_model]
token_ids: [T]
x: [T, d_model]
```

**Hints**

- Embedding lookup is row copying.
- For backward pass, gradients accumulate into rows used by tokens.

**Success criteria**

- Same token ID returns same embedding row.
- You can explain why embedding is equivalent to multiplying by a one-hot vector.

## 4.2 Linear layer

**Task**

Implement:

```text
y = xW + b
x: [B, in]
W: [in, out]
b: [out]
y: [B, out]
```

**Hints**

- Reuse matmul.
- Bias broadcasts across batch.

**Success criteria**

- Forward tests pass.
- You can write the shapes from memory.

## 4.3 Activation

**Task**

Implement one activation:

```text
tanh
ReLU
GELU later
```

**Hints**

- Start with ReLU or tanh.
- Save what you need for backward.

**Success criteria**

- Forward and backward tests pass for small values.

## 4.4 Tiny MLP language model

**Task**

Use a short context of previous tokens:

```text
embedding -> flatten -> linear -> activation -> linear -> logits
```

**Hints**

- Start with context length 2 or 4.
- Train on tiny text.

**Success criteria**

- Loss decreases below bigram on a small dataset.
- You can explain what extra information the context gives.

---

# Stage 5 — Transformer Inference From Scratch

Goal: implement a tiny decoder-only transformer forward pass manually.

## 5.1 Shape discipline for transformer tensors

**Task**

Before coding, write shape comments for:

```text
B = batch size
T = sequence length
C = model dimension
H = number of heads
D = head dimension = C / H
V = vocab size
```

Common shapes:

```text
tokens: [B, T]
x:      [B, T, C]
q,k,v:  [B, H, T, D]
scores: [B, H, T, T]
probs:  [B, H, T, T]
logits: [B, T, V]
```

**Success criteria**

- You can derive all attention shapes without looking.

## 5.2 Positional information

**Task**

Implement one:

```text
learned position embeddings
sinusoidal positions
RoPE later
```

**Hints**

- Learned position embeddings are easiest.
- Add token embedding + position embedding.

**Success criteria**

- Same token at different positions has different input vector.
- You can explain why a transformer needs position information.

## 5.3 Single-head causal attention

**Task**

Implement forward pass:

```text
q = xWq
k = xWk
v = xWv
scores = q k^T / sqrt(D)
mask future positions
probs = softmax(scores)
out = probs v
```

**Hints**

- Start with `B=1`, `H=1`, tiny `T` and `D`.
- Causal mask means position `t` cannot attend to positions `> t`.

**Success criteria**

- Future positions have probability zero.
- Shapes match at every step.
- You can explain attention as weighted averaging of values.

## 5.4 Multi-head attention

**Task**

Extend attention to multiple heads.

**Hints**

- Each head has dimension `D = C / H`.
- Concatenate heads back into `[B, T, C]`.

**Success criteria**

- Multi-head output shape is `[B, T, C]`.
- You can explain why multiple heads may learn different relations.

## 5.5 Normalization

**Task**

Implement RMSNorm first:

```text
rms = sqrt(mean(x_i^2) + eps)
y_i = weight_i * x_i / rms
```

**Hints**

- Test with simple vectors.
- LayerNorm can come later.

**Success criteria**

- RMSNorm output matches hand calculation.
- You can explain the purpose of normalization.

## 5.6 MLP block

**Task**

Implement transformer feed-forward block:

```text
linear -> activation -> linear
```

Llama-style later:

```text
silu(gate) * up -> down
```

**Success criteria**

- Input and output both have shape `[B, T, C]`.

## 5.7 Transformer block

**Task**

Combine:

```text
x = x + attention(rmsnorm(x))
x = x + mlp(rmsnorm(x))
```

**Hints**

- Residual connection requires same shape.

**Success criteria**

- One block forward pass runs on random tiny weights.
- You can explain pre-norm vs post-norm at a high level.

## 5.8 Tiny GPT inference

**Task**

Build:

```text
token embedding
position embedding
N transformer blocks
final norm
lm_head
sampling
```

**Success criteria**

- Random weights produce logits and sampleable tokens.
- All shapes are documented.

---

# Stage 6 — Manual Backprop and Training a Tiny Transformer

Goal: train a small transformer without PyTorch.

## 6.1 Backward pass for linear layer

**Task**

Implement gradients:

```text
y = xW + b
dx = dy W^T
dW = x^T dy
db = sum(dy)
```

**Success criteria**

- Compare against numerical finite-difference gradients on tiny inputs.
- You can explain each gradient shape.

## 6.2 Backward for embeddings

**Task**

Accumulate gradients into embedding rows used by token IDs.

**Success criteria**

- Repeated token IDs accumulate multiple gradient contributions.

## 6.3 Backward for activation and norm

**Task**

Implement backward for your chosen activation and RMSNorm.

**Hints**

- Use finite differences heavily.
- Keep dimensions tiny.

**Success criteria**

- Gradient checks pass within tolerance.

## 6.4 Backward for attention

**Task**

Implement attention backward in small pieces:

```text
out = probs v
probs = softmax(masked scores)
scores = q k^T / sqrt(D)
q,k,v = linear projections
```

**Hints**

- This is hard. Do not do it all at once.
- Keep `B=1`, `H=1`, `T=3`, `D=2` for tests.

**Success criteria**

- Finite-difference gradient checks pass on tiny cases.
- You can explain gradient flow through softmax.

## 6.5 SGD optimizer

**Task**

Implement:

```text
param -= learning_rate * grad
```

**Success criteria**

- Bigram, MLP, or transformer loss decreases.

## 6.6 AdamW optimizer

**Task**

Implement AdamW after SGD works.

**Hints**

- Maintain `m`, `v`, and timestep per parameter.
- Decoupled weight decay is separate from gradient.

**Success criteria**

- AdamW trains more smoothly than raw SGD on a small model.

## 6.7 Train first tiny transformer

**Task**

Train a very small character/byte-level transformer. This repo includes text data you can use after tiny hand-written strings work:

```text
datasets/tinyshakespeare.txt
```

Suggested first transformer scale:

```text
vocab_size: <= 256
context_length: 16-128
d_model: 32-128
layers: 1-4
heads: 1-4
```

**Success criteria**

- Loss decreases.
- Model overfits a tiny text file.
- Generated samples become recognizable.

---

# Stage 7 — Checkpoints, CLI, and Reproducibility

Goal: make experiments repeatable and usable.

## 7.1 Checkpoint format

**Task**

Save and load:

```text
magic/version
config
vocab/tokenizer metadata
parameter tensors
optimizer state optional
```

**Hints**

- Start with simple binary or JSON+binary.
- Include shapes and dtype.

**Success criteria**

- Save checkpoint, reload, get same logits for same input.

## 7.2 Train and infer commands

**Task**

Create commands like:

```text
zig build train -- --data input.txt --out model.bin
zig build infer -- --checkpoint model.bin --prompt "Once upon"
```

**Success criteria**

- Training and inference are separate commands.
- Inference works in a fresh process using only checkpoint data.

## 7.3 Experiment logging

**Task**

Log:

```text
step
train loss
validation loss optional
sample text
config
seed
```

**Success criteria**

- You can reproduce a run from its config and seed.

---

# Stage 8 — Llama-style Inference

Goal: learn practical LLM inference architecture.

## 8.1 Study `llama2.c`

**Task**

Read Karpathy's `llama2.c` and map each part to your code:

```text
config
weights
run state
forward pass
tokenizer
sampler
KV cache
```

**Success criteria**

- You can explain the full inference loop from token to next token.

## 8.2 Port a minimal Llama-style forward pass to Zig

**Task**

Implement the forward pass for a tiny Llama-like checkpoint format.

**Hints**

- Start with f32.
- Start without quantization.
- Add RoPE and KV cache carefully.

**Success criteria**

- Loads known tiny weights or generated toy weights.
- Produces stable deterministic logits.

## 8.3 KV cache

**Task**

Implement and test key/value cache for autoregressive generation.

**Hints**

- Without cache: recompute all previous tokens.
- With cache: append new K,V per step.

**Success criteria**

- Cached and uncached outputs match for the same prompt.
- You can explain why cache improves inference speed.

---

# Stage 9 — ZML Second Path

Goal: re-express understood models using Zig-native compiled tensor execution.

Important: ZML is not a replacement for understanding. Use it after raw implementations.

## 9.1 Complete ZML first model tutorial

**Task**

Follow ZML's `write_first_model.md` tutorial and run the simple layer.

**Concepts to notice**

```text
model as Zig struct
forward() as tensor graph
shape/dtype tensor definitions
compile before execution
bufferized weights and inputs
explicit result copy-back
```

**Success criteria**

- You can explain what is known at compile time vs runtime.
- You can explain Tensor vs Buffer.

## 9.2 Rebuild your Linear layer in ZML

**Task**

Implement your earlier raw Zig linear layer as a ZML module.

**Success criteria**

- Raw Zig output and ZML output match on tiny data.

## 9.3 Rebuild bigram or MLP inference in ZML

**Task**

Port a previously understood model to ZML.

**Success criteria**

- You can compare raw and ZML outputs.
- You can explain what ZML handles for you.

## 9.4 Explore transformer inference in ZML

**Task**

Only after raw attention works, express attention/transformer forward in ZML.

**Success criteria**

- One tiny transformer block runs through ZML.
- You can still explain every operation, not just call APIs.

---

# Stage 10 — Performance and Systems Work

Goal: make correct code faster and more production-like.

Do not start here.

## 10.1 Profiling

**Task**

Measure time spent in:

```text
matmul
softmax
attention
sampling
tokenization
```

**Success criteria**

- Optimization decisions are based on measurements.

## 10.2 Cache-friendly matmul

**Task**

Improve matmul locality with loop order, blocking, or transposed weights.

**Success criteria**

- Faster than naive version on benchmark.
- Same outputs within tolerance.

## 10.3 SIMD and threading

**Task**

Add SIMD/threading only after profiling.

**Success criteria**

- Deterministic correctness tests still pass.
- Speedup is measured.

## 10.4 Quantization

**Task**

Implement simple int8 or group quantization for inference weights.

**Success criteria**

- Memory decreases.
- Output degradation is measured.

## 10.5 GPU / accelerator path

**Task**

Explore ZML or direct GPU kernels after CPU baseline is correct.

**Success criteria**

- CPU and accelerator outputs match on tiny cases.

---

# Recurring Drills

Use these throughout the journey.

## Shape drill

For every function, write input and output shapes before coding.

**Pass condition**: no mystery dimensions.

## Tiny-number drill

Before random tests, create one hand-computed example.

**Pass condition**: you can compute expected output with paper/math.

## Explain-it-back drill

After implementing a task, explain:

1. What does this function compute?
2. What are the shapes?
3. Where can it fail numerically?
4. How do we test it?
5. How is it used in a language model?

## Rewrite drill

After a break, rewrite a primitive from memory:

```text
matmul
softmax
cross entropy
bigram gradient
linear backward
attention forward
```

**Pass condition**: you can recreate it without copying.

---

# Pointers

Use these references at the right time:

- Karpathy `llama2.c`: simple practical Llama-style inference.
- Karpathy `nanoGPT`: architecture and training loop reference, despite PyTorch.
- Karpathy `nanochat`: later end-to-end chat/data/tokenization ideas.
- CogitatorTech `zigformer`: Zig-native transformer reference.
- ZML tutorial `write_first_model.md`: second-path compiled tensor execution in Zig.

Remember: references are for checking understanding, not skipping the work.
