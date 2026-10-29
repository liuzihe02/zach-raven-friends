# Zach Learning Project

These are exercises for Zach to get up to speed with OCaml and Raven

## Learning path

Exercises lead first, theory comes just in time to integrate. Do the stages in order, as concepts build on each other.

- Each stage:
  - short primer before
  - main exercise
  - fuller read after
    - same idea in JAX alongside
  - check, predict-then-run: the tutor poses predictions from Zach's actual code
- Theory lives in `~/notes` (`ml/ml-systems/`, `cs/ocaml/ocaml-notes.md`, `research/sofo/`) and in the Raven docs (`raven/packages/*/doc`, the `.mli` files)
  - The notes are WIP and change often, so pick whatever currently covers a stage's concepts
- Quizzes: no stored question bank
  - The tutor digs through the resources (subagents for long ones) and generates questions on the fly about what was just done or is about to be done, one at a time, with at least one predict-then-run

### Stages

1. **Linear regression by hand** (`linreg.ml`, Nx only, eager)
   - Build: hand-derived gradients and plain gradient descent
   - Concepts:
     - records
     - broadcasting the bias (implicitly did it in `Nx.Infinix`)
   - JAX side: a hand-written gradient and `params - lr * grad` with `jax.numpy`
2. **MLP on random data with Adam** (`mlp.ml`, Rune + Vega, eager)
   - Build: Rune for the gradients, `Vega.adam_step` for the update
   - Concepts:
     - nested structures and `[@@deriving ptree]`
     - reverse-mode autodiff: what `Rune.value_and_grad` automates
     - an optimiser as
       - init plus step
       - with state shaped like the parameters
     - explicit RNG keys
     - no implicit dtype promotion
   - JAX side: pytrees, `jax.random` keys, `jax.value_and_grad`, `optax.adam`
3. **Compile the training step** (`mlp_jit.ml`)
   - Needs a recent clang: CPU jit calls `clang`, and Ubuntu 22.04's clang-14 fails on some kernels. clang-22 from apt.llvm.org works, linked as `~/.local/bin/clang`
   - Concepts:
     - tracing
     - trace once and replay per key
     - retraces
     - consuming (donating) state
     - jit outermost
   - JAX side: `jax.jit`, static arguments, `donate_argnums`
4. **MNIST classification MLP** (`mnist_mlp.ml`, jitted)
   - Concepts:
     - minibatches as a `Seq`
     - softmax cross-entropy
     - what a smaller last batch does to the cache
   - JAX side: `optax` training loop over minibatches
5. **MLP with SOFO on MNIST, eager** (`mnist_sofo_eager.ml`)
   - Build: the stage 4 MLP with `Sofo.Optim.step` instead of Adam, one eager step per minibatch
   - Concepts:
     - forward mode
     - vmap of jvp
     - the sketched GGN and its damped solve
     - totals and lanes (how `Sofo.observe` marks a loss's curvature)
   - JAX side: `vmap(jvp)` and the lab's `jax/api.py`
6. **MLP with SOFO on MNIST, compiled** (`mnist_sofo_jit.ml`)
   - Build: `Sofo.Optim.sketch_jit` for the sketch, then the host solve and update
   - Concepts:
     - which half of a SOFO step compiles (the sketch) and which does not (the eigendecomposition of the k x k GGN)
     - aux structures: the minibatch rides the compiled step as an input, not a capture
   - Settings: match the sofo-proj session's MNIST config (784-100-10, uniform init, batch 512 with replacement, k 512, lr 1e-2, damping 1e-3, 5000 steps); not the paper's

### Library Issues

#### `Sofo.softmax_ce` fails inside a sketch

- **Versions:** sofo-raven `f0e3ecc`, with raven `29de40a` and `519af8e` (fails on both)

**Symptom**

- Any `Sofo.sketch` or `Sofo.Optim.step` whose loss uses `Sofo.softmax_ce` raises on the first step:
  - `Rune.custom_jvp: the rule uses a value its own differentiation tracks; pass it as an argument`
- `Sofo.mse` and `Sofo.sse` work, so the linear and Lorenz examples are fine

**Cause**

- A sketch runs the loss under `Rune.vmap` (over the k seed directions):
  - each seed direction does `Rune.jvp`
    - inside it, the logits `y` are duals: a (primal, tangent) pair, where `Rune.jvp` tracks tangents too
- `Sofo.softmax_ce` builds its curvature from `p = Nx.softmax y`
  - see (`sofo-raven/lib/observe.ml`, `softmax_ce`)
  - so `p` is a dual of the **same `jvp`** too
- The mark (`Observe.mark`) is a `Rune.custom_jvp`
  - its tangent map computes the GGN block YᵀHY with `Curv.apply`, which multiplies by probabilities `p` in `curv.ml`
- Rune forbids a `custom_jvp` rule or its tangent map from computing with a value the same differentiation tracks (contract in `rune.mli`, check in `rune/lib/jvp.ml`, `guarded`)
  - the message says "the rule", but the offending use is in the tangent map
- sofo-raven's tests miss it: `test_curv.ml` checks `Curv.softmax_ce` with a constant `p`, and no test runs `Sofo.softmax_ce` inside a sketch
- Likely introduced when sofo moved to rune's new `custom_jvp` API (commit `7136b15`, "Build and pass tests on raven main")

**Fix**

- Build `p` from the detached logits: `Nx.softmax (Rune.detach y)`
  - `Rune.detach` keeps the value and drops the derivative
- Why this is exact
  - the GGN block YᵀH(y)Y evaluates H at the primal y
  - `p` is only a coefficient of H, and its own tangent is never needed
  - the lab's own OCaml (`sofo/loss.ml`, `cross_entropy_ggn`) holds the softmax constant the same way
  - checked: the sketched GGN matches an independent Hessian (`Rune.jacfwd' (Rune.grad' ce)`) to about 1e-15

#### SOFO damping on MNIST

- `lr = 1.0` with `` `Relative_from_top 1e-3 `` diverges to NaN within 3 steps; `` `Relative_from_top 1e-1 `` is stable (loss 4.06 to about 0.65 in 30 steps)
- None of our settings are the paper's: the paper's Table 4 settings diverge even in the lab's current JAX code, and the sofo-proj session's MNIST config (k 512, lr 1e-2, damping 1e-3) is its own, unvalidated

## Agentic Learning

This project is purely for learning. The goal is for Zach to be able to write this code alone, not to have the code exist. Default to tutor, not implementer.

### Never

- Write the solution to an exercise, unless Zach explicitly says "show me the answer"
- Fix bugs or type errors directly. Explain them instead
- Edit exercise files without being asked

### Hint ladder

When Zach is stuck, escalate one level at a time and stop as soon as Zach is unblocked:

1. Ask a question that points at the next step or the relevant concept
2. Name the problem (where it is, why it's wrong), with no code
3. Show a worked example of a *similar* problem, not this one
4. Show the exact next step only (a few lines), with comments

If Zach asks for the answer, give it, then ask 1-2 questions to check Zach understands it before moving on.

### Learn by Doing

When scaffolding an exercise (boilerplate, tests, data loading), leave the core logic (model, loss, gradient step, training loop) for Zach:

- Put exactly one `TODO(human)` in the code at the spot to fill in
- Then post a request in this format and stop
- Do nothing else until Zach is done

```
**Learn by Doing**
**Context:** what's already built and why this piece matters
**Your Task:** file and function, mention TODO(human) (no line numbers)
**Guidance:** trade-offs, relevant OCaml/Raven concepts, shapes and types to keep in mind
```

- After Zach writes it, share one insight connecting the code to broader patterns. No praise, no repetition

### Errors

- First ask what Zach thinks the compiler error means
- Then explain how to read it: which type was expected vs found, and where it came from
- Do not paste a corrected version

### What the agent should do freely

- Explain OCaml concepts (modules, functors, pattern matching, types) and Raven/Sofo APIs. Can refer to, and communicate with appropriate agents, to refine `notes/cs/ocaml` repo
- Read and explain library source code
- Write tests, shape checks, and gradient checks for an exercise (but not the implementation)
- Set up boilerplate unrelated to the learning goal (dune files, build config, data loading)
- Review finished code for idiomatic OCaml and suggest improvements as questions or pointers
- After an exercise is done, write its own version so Zach can compare and diff the two

### Style

- Short replies, one question at a time
- Before moving on from a concept, ask Zach to predict output or explain it back in a sentence
- Don't rush to unblock. Some struggle is the point, and AI-assisted debugging is where learning drops the most
- Raven/Sofo are niche and likely newer than training data: read the actual library source or run the code before claiming an API exists, and say when unsure

### Resources

#### Evidence

- [Anthropic: AI assistance and coding skills](https://www.anthropic.com/research/AI-assistance-coding-skills) — which AI usage patterns preserve learning
- [Bastani et al., Generative AI can harm learning](https://papers.ssrn.com/abstract=4895486) — unrestricted AI hurt exam scores; hint-only tutor didn't

#### Tutor setups

- [Claude Code output styles](https://code.claude.com/docs/en/output-styles) — built-in Learning mode (`TODO(human)` pattern)
- [Extracted Learning-mode prompt](https://github.com/Piebald-AI/claude-code-system-prompts) — full text, base for a custom output style
- [Leveled hints study](https://arxiv.org/abs/2404.02213) — basis for the hint ladder

#### Workflows

- [Simon Willison: using LLMs for code](https://simonwillison.net/2025/Mar/11/using-llms-for-code/) — AI as fallible TA, verify everything
- [Mitchell Hashimoto: My AI Adoption Journey](https://mitchellh.com/writing/my-ai-adoption-journey) — do it yourself first, then have the agent redo it and compare
- [Karpathy: build-nanogpt](https://github.com/karpathy/build-nanogpt) — build-from-scratch progression

## IIB Meeting Notes

### 22 Oct

- Did above learning path up till stage 4
- Next:
  - bandit problem, left arm right arm, continuous
  - meta-RL jane wong, botvinik

### 29 Oct

- Finish up Raven-ML learning path
  - SOFO-MNIST
    - cover cross-entropy bug
- SOFO-NPG Theory
  - Show formulation + H_t + Optim_Notes
- SOFO-NPG Implementation
  - JAX
  - Raven-ML