# neural-net

A from-scratch C++ neural network library. No dependencies beyond the standard
library and SFML (visualizer only — see `src/viz/`, which is not part of the
goal).

## What this project is for

This is an **educational project**. The goal is to re-implement the parts of
PyTorch that matter — a strided tensor, broadcasting, batched matmul, autograd —
well enough to then build real models on top of them: CNNs, a GPT, and whatever
comes after.

That framing decides most arguments:

- **Understanding beats convenience.** Pulling in Eigen or libtorch would defeat
  the point. Linking a BLAS for `sgemm` later is fine, because by then the naive
  version exists and can be compared against.
- **Generality is earned, not assumed.** Build for the models actually on the
  roadmap. Don't add an abstraction because a real framework has one.
- **But don't paint into corners.** Where a design choice would be expensive to
  reverse and the roadmap clearly needs the general version, take the general
  version now. The `dim` parameter on `softmax` is the standing example: an MLP
  never needs it, attention does.

Brian writes the code. Explain proposed changes and show them before applying;
don't hand over finished implementations unasked. Comments in the source
document non-obvious invariants only — never restate a chat explanation in a
comment.

**Exception: `src/viz/` is yours.** The visualizer is a debugging aid, not part
of the goal. Change it freely, without walking through it first — just keep it
compiling and keep it off the hot path. It gets dragged along by every refactor;
don't spend Brian's review time on it.

## Current state

Working and tested end to end: an MLP trains on MNIST through the `Tensor` path
(`{784,128,64,10}` reaches ~88% in 3 epochs on a 6k subset).

`Matrix` / `MatrixOps` are the previous implementation. They still compile and
`Metrics.h` still carries an `argmax(const Matrix&)` overload, but nothing on the
live path uses them. They are dead code kept deliberately for now.

`src/math/TODO.md` tracks Tensor-level work item by item.

---

## Tensor

### Storage and views

`Tensor` is metadata over a shared buffer:

```cpp
std::shared_ptr<std::vector<float>> data;
int64_t offset;  int ndim;  int64_t numel;  bool contiguous;
std::array<int, MAX_DIMS> shapes;
std::array<int64_t, MAX_DIMS> strides;
```

Address of any element is `offset + Σ strides[d] * idx[d]`. Strides are flat
absolute element offsets — there is no nesting.

- **Fixed `std::array` sized `MAX_DIMS = 8`, not `std::vector`.** Shape metadata
  is touched constantly; heap-allocating it per tensor would dominate the cost of
  cheap view operations.
- **Views share the buffer, copies are deep.** `slice` and `transpose` return a
  new `Tensor` pointing at the same `shared_ptr`. `Tensor x = y;` allocates and
  copies. The reasoning: aliasing already has names at the call site, plain
  assignment says nothing, so assignment should do the unsurprising thing.
- **A copy takes the source's shape but not its strides.** Copying a transposed
  view gives a plain row-major tensor of the transposed shape, at offset 0. This
  doubles as "make contiguous".
- **Rule of five is written out.** Declaring a copy constructor suppresses the
  implicit moves, so the moves are explicitly `= default`. Without them every
  non-elided return would deep-copy.
- **`clone()` deliberately does not exist** — the copy constructor is it.

### Contiguity

`contiguous` is computed in both constructors by walking strides right to left
against what row-major would generate. The `shapes[d] != 1` guard matters: a
size-1 dim can carry any stride, since it is never indexed past 0.

**Contiguous does not mean offset 0.** A contiguous view can begin partway into
the buffer. `addressOf`'s fast path is `offset + n`, not `n`.

### `addressOf`

Every whole-tensor walk goes through `int64_t addressOf(int64_t n)`: flat element
number to buffer index, with the contiguous fast path inside the helper rather
than at each call site, so no caller can forget it.

### Element access

`at` / `setAt` take `const std::vector<int>&` and keep `data->at(i)` bounds
checking. Everything else uses unchecked `(*data)[i]`.

That inconsistency is intentional: the `addressOf` walks compute their own
indices from `numel`, so an out-of-range there is a stride bug in `Tensor`.
`at`/`setAt` index off caller-supplied coordinates with no validation yet — the
two validation asserts are still open on the TODO, and until they land the bounds
check is the only guard. It is also free; measured, it costs nothing.

Measured cost of `at`: ~7.5 ns per call, of which ~5 ns is constructing the
`std::vector<int>` temporary. An `initializer_list` overload would fix that and
was deliberately deferred — nothing on a hot path calls `at`, and hot paths
should be getting raw access through `TensorOps` instead.

---

## The `Tensor` / `TensorOps` boundary

`friend struct TensorOps;` — one declaration covering every op, present and
future.

The rule for which side an operation lands on:

| | |
|---|---|
| **`Tensor` member** | metadata-only: `slice`, `transpose`, later `permute`, `narrow`, `reshape`. Plus arithmetic simple enough to go through `addressOf`. |
| **`TensorOps` static** | anything that walks elements at scale and wants raw `data` / `strides` / `offset` without per-element overhead: `matmul`, activations, later conv and reductions. |

**Keep this boundary tight.** Nothing outside `Tensor.cpp` and `TensorOps.cpp`
touches `data` directly. That discipline is what keeps a future GPU port to two
files instead of ten.

---

## matmul

**The last two dims are the matrix; everything to the left is batch.** One kernel
covers every rank — 2D is just the case where the batch count is 1. There is no
per-rank branching.

- Batch ranks may differ. They are **right-aligned**, the shorter padded with
  implicit 1s, numpy's rule.
- A broadcast dim gets **stride 0**, so the odometer re-reads the same matrix
  with no copy and no branch in the loop. "Missing dim" and "size-1 dim" collapse
  into the same zero.
- 1D operands are **promoted then squeezed**: `(K)` becomes `(1,K)` on the left,
  `(K,1)` on the right, and the invented dim is sliced back off. `(K)×(K)` comes
  out as a 1-element dot product rather than a 1×1 matrix.
- Loop order is **ikj**, not ijk. Measured 1.8× at 128, 4.8× at 512, 10.8× at
  1024. With `j` innermost both `B` and `O` are walked along their last dim and
  `A[i][k]` lifts out of the loop.
  **Caveat:** ikj assumes `bs1 == 1`. A transposed right operand defeats it and
  ijk would be better there. Correctness is unaffected. Open on the TODO.

Multi-batch-dim support is not speculative — `(batch, heads, seq, dk) × (batch,
heads, dk, seq)` is the attention shape, and it already works.

---

## Activations

- `ReLU` / `ReLUPrime` are elementwise maps through `addressOf`, so they work on
  views for free.
- **`ReLUPrime` is a step, not a clamp.** Every positive input derives to 1
  regardless of magnitude. The derivative at exactly 0 is 0, matching `Matrix`.
- **`softmax` takes an explicit `dim`.** The two-arg form is the real one; the
  one-arg form exists only for the current column-vector layout and asserts it.
- **There is deliberately no `softmaxPrime`.** Softmax is not elementwise — its
  Jacobian is `p_i(δ_ij − p_j)`, a full matrix. Chained with cross-entropy the
  whole thing collapses to `p − y`, which is what `crossEntropyPrime` returns and
  why `Layer::backward` passes the softmax gradient straight through.
  **This is only valid for softmax paired with cross-entropy.** Pairing softmax
  with MSE would silently produce wrong gradients.

---

## Traps

Things that compile and are wrong, or that bit us once already:

- **`operator*` on `Tensor` is hadamard, not matmul.** On `Matrix` it was matmul.
  `W * input` compiles and then asserts at runtime on mismatched shapes. Use
  `TensorOps::matmul`.
- **The save/load header writes `in` then `out`.** `W` is `(out, in)`, so `save`
  emits `dims[1], dims[0]`. Flip it and every saved network silently reloads
  transposed.
- **`getDims()` returns by value.** Hoist it out of loops; leaving it in a loop
  condition allocates a vector per iteration.
- **A `Layer` holds seven Tensors.** Copying one deep-copies all seven. Use
  `std::move` when pushing into a container.

---

## Decided but not yet built

These are settled design decisions, not open questions. Implement them this way.

### Layout: batch-leading, `(batch, features)`

The current column-vector layout `(features, 1)` is a dead end. CNNs are
`(N,C,H,W)` and attention is `(batch, heads, seq, dk)` — both put batch first and
both need multiple batch dims, which is exactly what `matmul` already targets.

Under `(batch, features)` with weights stored `(in, out)`:

```
forward:  matmul(x (N,in),  W (in,out))   ->  (N, out)
dW:       matmul(xᵀ (in,N), err (N,out))  ->  (in, out)   batch contracted, no reduction
db:       column sum of err (N,out)       ->  (out,)
```

`xᵀ` is a free view, so both the forward and the weight gradient are single
GEMMs. Consequences: `W` flips orientation, the save format changes, the
column-vector convention dies, `softmax`'s one-arg overload is removed, and
`reshape` becomes load-bearing (conv output feeding a linear layer).

### Broadcasting and reduce-sum are one feature

The backward of a broadcast is a **sum over the broadcast dims**. If
`(N,out) + (out,)` broadcasts, the bias gradient is `sum over N`. Do not add
elementwise broadcasting without adding the reduction that reverses it.

The right-aligned compatibility rule currently lives inside `matmul`. Extract it
so the elementwise ops share one implementation.

### Autograd: a tape, layered on top of `Tensor`

```
Tensor  — strided buffer, no autograd knowledge
Var     — shared_ptr<Node>; Node holds value, grad, parents, backward closure
```

**Do not merge grad machinery into `Tensor`.** Keeping `Tensor` a plain data
container is what keeps the GPU work orthogonal and stops every intermediate view
from carrying grad state.

Recording ops linearly means reverse iteration is already topologically valid —
no sort needed. Gradients accumulate (`+=`), since a tensor used twice gets two
contributions.

Two properties of the existing code make this work, and must be preserved:

- **Ops never mutate their inputs.** Every operator returns a fresh Tensor. If an
  input could be overwritten after being recorded, the tape would need version
  counters.
- **`shared_ptr` storage** lets the tape hold intermediates alive by refcount
  rather than by copying.

Every op needs a registered backward, including the views — `transpose`'s
backward is `transpose`, `slice`'s is scatter-into-zeros.

General autograd is not in tension with hand-written backward. Register a fused
backward for the specific spots where it avoids materializing a large
intermediate — softmax+cross-entropy already is one.

### GPU: after batching, not before

The blocker is not the kernel, it's arithmetic intensity. At batch 1 every matmul
is a matvec and the GPU sits idle; batching is a hard prerequisite.

The kernel itself is one `cublasSgemm` call. The actual work is that **weights
must stay device-resident** — `applyGradients` mutates `W` every batch, so if `W`
lives on the device then `operator-` and `operator*(float)` must too, which pulls
in the rest of the elementwise ops.

That means storage becomes `shared_ptr<float>` with a `cudaFree` deleter plus a
residency flag — a change in `Tensor`, not `TensorOps`.

**Views come free.** `transpose`, `slice`, offsets and strides are pure metadata
and already work unchanged on a device tensor. That is normally the fussy part of
a GPU tensor library.

Worth linking OpenBLAS first: it's a few hours, typically 10–50× over the naive
`ikj`, and it proves the backend seam works before committing to CUDA.

---

## Roadmap

1. Batch-leading layout
2. Broadcasting **+** its reduce-sum
3. Autograd
4. CNN — im2col turns convolution into the existing `matmul`, so it's a shape
   transform plus pooling, not a new kernel
5. GPT — LayerNorm, GELU, embedding gather, causal masking. The hard part is
   backward through attention and the residual branches, which is why autograd
   comes first

GPU is orthogonal and can slot in any time after step 1.

Autograd lands before CNN deliberately — otherwise conv backward gets
hand-derived and then thrown away.

---

## Build

```
make          # -std=c++20 -Wall -Wextra -g -O2, sources globbed recursively
./nn          # TRAIN in src/app/Config.h picks train vs visualize
```

New `.cpp` files need no Makefile change. Keep the build warning-clean.
