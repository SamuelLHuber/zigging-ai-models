# Zigging AI Models

Learn LLM inference and tiny-model training from scratch in Zig — no PyTorch, no magic first, just manual math, tests, repetition, and understanding.

Start here: **[JOURNEY.md](./JOURNEY.md)**

## What this is

This repository is a guided curriculum and codebase for building up from raw Zig loops to small language models:

- tensor and matrix primitives
- softmax, cross entropy, sampling
- character/byte tokenization
- a trainable bigram language model
- an MLP language model
- tiny transformer inference
- manual backprop and tiny transformer training
- Llama-style inference concepts
- eventually, a second implementation path with ZML

The point is not to get a polished model quickly. The point is to understand every major piece well enough to rebuild it.

## Learning style

Early on, the student should type the core implementations manually. An assistant should first give concise motivation and background for why a task matters, then provide hints, tests, shape checks, and review, but should not simply paste full solutions for foundational tasks.

We avoid PyTorch initially because we do not want a large dependency stack in the critical path, we want to reduce security exposure from Python packages and package registries, and we want full control over memory, execution, shapes, and model internals. PyTorch and other frameworks may be used later as conceptual references, but not as the first implementation path.

Allowed references: Zig docs, Zig standard library docs, compiler errors, this curriculum, small hand-written math notes, and conceptual explanations. Later, after implementing the relevant primitive ourselves, we may compare against references such as Karpathy's `nanoGPT`, `llama2.c`, or ZML docs. Early cheating means copying full implementations of core primitives such as matmul, softmax, cross entropy, training loops, or attention before attempting them.

We optimize for:

- learning by doing
- tiny hand-checkable tests
- explicit tensor shapes
- repetition
- correctness before speed
- CPU `f32` before GPU, SIMD, quantization, or `f16`

## Getting started

This repository begins as a normal Zig project:

```bash
zig build test
zig build run
```

### Clean/reproducible dev environment

If you want to keep your local system clean, use the included [devenv](https://github.com/cachix/devenv) setup instead of installing tools globally:

```bash
devenv shell
zig build test
```

The dev shell provides Zig, Jujutsu, Git, and the GitHub CLI. You can also use the devenv setup as the basis for a Docker image if you prefer containerized development.

If starting a fresh copy without version control, initialize with:

```bash
jj git init --colocate .
zig init
```

Then follow the curriculum in [JOURNEY.md](./JOURNEY.md).

## Helpful references

Use references to check understanding, not to skip the work.

- [Karpathy: Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — excellent Python/PyTorch learning companion for GPT concepts.
- [karpathy/llama2.c](https://github.com/karpathy/llama2.c) — simple practical Llama-style inference in C.
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — compact GPT reference: roughly ~300 lines for the training code and another ~300 lines for the model definition, making it excellent for reading end-to-end.
- [karpathy/nanochat](https://github.com/karpathy/nanochat) — later-stage end-to-end chat/model pipeline reference.
- [CogitatorTech/zigformer](https://github.com/CogitatorTech/zigformer) — Zig-native transformer reference.
- [zml/zml](https://github.com/zml/zml/) — high-performance, production-oriented AI/ML infrastructure in Zig. Read its getting started docs and code when you want to see how serious Zig AI systems structure models, compilation, buffers, and accelerator execution.
- [ZML first model tutorial](https://github.com/zml/zml/blob/master/docs/tutorials/write_first_model.md) — Zig-native compiled tensor/model execution after raw primitives are understood.

## Included dataset

This repo includes Karpathy's Tiny Shakespeare text dataset:

```text
datasets/tinyshakespeare.txt
```

Use it for the early bigram, MLP, and tiny transformer language-model training tasks once your code works on very small hand-written strings.

## First milestone

Train a tiny character-level bigram model that first overfits a tiny hand-written string, then trains on `datasets/tinyshakespeare.txt` and generates recognizable text.

That is the first real “I trained my own model” moment.
