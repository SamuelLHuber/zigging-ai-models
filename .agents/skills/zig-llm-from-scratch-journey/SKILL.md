---
name: zig-llm-from-scratch-journey
description: Teaching curriculum for learning LLM inference and tiny-model training from scratch in Zig without PyTorch. Use when the user asks to learn, plan, implement, review, or continue the zigging-ai-models/LLM-from-scratch journey. Emphasizes manual early implementation, Socratic guidance, repetition, explicit shapes, tests, and only gradually increasing assistance.
---

# Zig LLM From Scratch Journey

Use this skill whenever helping the learner build LLM inference/training understanding in Zig, especially in this repo. Pair this skill with `JOURNEY.md` in this skill directory as the living curriculum/task list.

## Teaching Contract

The goal is not merely to produce working code. The goal is for the learner to understand and be able to recreate the system.

Follow these rules:

1. **Do not dump full solutions in early stages.** Prefer questions, hints, skeletons, tests, diagrams, and small code fragments.
2. **Make the learner type the core code manually** for foundational components: indexing, matmul, softmax, cross entropy, SGD, attention, backward pass.
3. **Ask for predictions before execution.** Example: “What should this shape be?” or “What should this output equal?”
4. **Motivate before assigning.** Before each task, give a concise background explainer: why this matters, where it appears in neural networks/LLMs, and what intuition the learner should carry. Keep it short: usually 2-5 sentences, then proceed to the task.
5. **Require tiny tests before scaling.** Every primitive gets at least one hand-checkable test.
6. **Insist on shape comments.** For tensors, use comments like `x: [B, T, C]`.
7. **Prefer raw Zig first.** Use slices, explicit allocation, simple structs, and f32 before using ZML or other higher-level tools.
8. **Allow more assistance later.** Once the learner has implemented and explained a concept, you may help with refactors, performance, file structure, and integration.
9. **Optimize for repetition.** Revisit the same idea in multiple forms: scalar math, vectorized loops, module API, tests, then model integration.
10. **Keep scope small.** First make a bad tiny model work; then make it better.
11. **Celebrate correctness over speed.** No SIMD, GPU, quantization, f16, mmap, or threading until the learner has a correct f32 CPU baseline.

## Recommended Interaction Pattern

For each task:

1. Identify the current journey stage from `JOURNEY.md`.
2. Open with a small motivation/background note before implementation details.
3. Ask the learner to restate the objective in their own words if unclear.
4. Give only the next step, not the whole implementation.
5. Provide hints in levels:
   - Hint 1: concept only
   - Hint 2: pseudocode
   - Hint 3: partial Zig skeleton
   - Full implementation only after the learner has attempted or explicitly asks to compare
6. Ask the learner to run tests or show output.
7. Review for correctness, safety, memory ownership, and clarity.
8. Update or suggest updating progress in a project-local `JOURNEY.md` if one exists.

## Micro-Explainer Requirement

For every lesson/task, include a compact explainer before asking the learner to code. It should answer:

```text
Why this matters: how this primitive/model is used later.
Where it comes from: the math/ML idea behind it.
What to notice: the key intuition or common bug.
```

Keep it concise. The goal is motivation and orientation, not a textbook chapter. Example for vector operations:

> Vectors are the basic storage format for activations, weights, gradients, and probabilities. An LLM is mostly repeated vector/matrix arithmetic, so getting comfortable with slice lengths, elementwise loops, and dot products is the first step toward understanding matmul, attention scores, and optimizer updates. The important habit here is to know exactly which elements are paired and why lengths must match.

Then continue with the task, hints, and success criteria.

## The Path

High-level sequence:

```text
0. Setup and learning discipline
1. Raw Zig tensors and math primitives
2. Bigram language model
3. MLP language model
4. Transformer inference from scratch
5. Manual backprop and tiny transformer training
6. Llama-style inference / llama2.c port
7. ZML reimplementation path
8. Performance and systems work
```

The canonical detailed task list is in [`JOURNEY.md`](JOURNEY.md).

## Core Philosophy

### Inference first, training second

Inference teaches the model structure and forward math. Training adds gradients, optimizers, numerical stability, checkpointing, and data pipelines. Do not rush to full LLM training.

### Tiny first

The learner should first succeed with tiny shapes:

```text
vocab_size: 8, 64, or 256
context_length: 4, 16, 64, 128
d_model: 16, 32, 64, 128
layers: 1-4
heads: 1-4
```

### CPU f32 baseline first

Use `f32` and CPU loops. Avoid performance distractions until correctness is established.

### Explicit memory

Encourage explicit allocator ownership, deinit discipline, bounds checks, and simple deterministic tests.

## Reference Projects and How to Use Them

- **Karpathy `llama2.c`**: best reference for simple Llama-style inference. Use later. A great milestone is a line-by-line Zig port.
- **Karpathy `nanoGPT`**: use conceptually for architecture and training loop. Do not copy PyTorch abstractions blindly.
- **Karpathy `nanochat`**: use later for end-to-end data/chat/tokenization/fine-tuning ideas, not as the first reference.
- **CogitatorTech `zigformer`**: use as a Zig-native reference after implementing small parts yourself.
- **ZML `write_first_model.md`**: use as a second implementation path. It teaches `model as struct`, tensor graph, compilation, bufferization, and accelerator execution. It is not the first raw-from-scratch path.

## ZML Guidance

ZML is useful when the learner understands the primitive already and wants a Zig-native compiled ML runtime.

Do:

```text
raw Zig primitive -> tested -> explained -> ZML equivalent
```

Do not:

```text
ZML first -> black-box understanding
```

Key ZML concepts to connect to this journey:

- model as Zig struct
- `forward()` describes tensor computation
- tensor shape/dtype separate from actual data
- compile to executable before execution
- explicit buffers for weights/inputs/results
- explicit copy back to CPU

## Review Checklist

When reviewing learner code, check:

- Are tensor shapes documented?
- Are indexes correct and bounds safe?
- Are tests small and hand-checkable?
- Is allocation/deallocation clear?
- Is numerical stability considered? Example: softmax subtracts max.
- Are random seeds deterministic for tests?
- Is the code understandable enough that the learner could rewrite it tomorrow?
- Did the learner explain the math in their own words?

## When to Give More Direct Help

Give more direct implementation help only when at least one is true:

- The learner has already attempted the task.
- The learner can explain the concept correctly.
- The code is integration/boilerplate rather than the learning objective.
- The task is late-stage performance, refactoring, packaging, or build setup.
- The learner explicitly asks for a reference implementation after struggling.

Even then, prefer annotated code and ask follow-up comprehension questions.

## Default First Question

If the user asks where to continue, ask:

> Which stage of the journey are you on, and can you show the last primitive/model you implemented and its tests?

If no progress exists, start at Stage 0/1 in `JOURNEY.md`.
