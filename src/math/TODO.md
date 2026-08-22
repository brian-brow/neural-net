# Tensor TODO

## State
- [x] `offset` member, applied in `at` / `setAt`
- [x] `contiguous` flag

## Views
- [x] `transpose()` — last two dims
- [ ] `permute(const std::vector<int>&)`
- [ ] `reshape(const std::vector<int>&)`
- [x] `slice(int dim, int index)`
- [ ] `narrow(int dim, int start, int len)` — range, keeps rank

## Non-contiguous correctness
- [x] `print` walks logically via `offset` + `strides`
- [x] `zero`, `randomize`, `operator+`, `operator-`, `operator*` go through `addressOf`
- [x] contiguous fast path, strided walk otherwise (both inside `addressOf`)

## Broadcasting
- [x] shape compatibility: align right, each dim must match or be 1 — batch dims in `matmul` only
- [ ] extract that rule out of `matmul` so the elementwise ops can share it
- [ ] replace `assert(shapes == other.shapes)` in `+`, `-`, `*`

## TensorOps
- [x] `matmul` — last two dims are the matrix, everything left is batch
- [x] mismatched batch rank, right-aligned, broadcast dims get stride 0
- [x] 1D operands — promote to `(1,K)` / `(K,1)` then squeeze
- [x] `ikj` loop order — 10x at 1024, but assumes `bs1 == 1`
- [ ] transposed right operand defeats `ikj` — pick order from `bs1`, or pack `B`

## Access for TensorOps
- [x] `getNumel()`
- [x] `getStrides()`
- [x] `getOffset()`
- [x] raw buffer access or `friend` — `friend struct TensorOps`

## Types
- [x] `numel` to `int64_t`
- [x] `offset` to `int64_t`
- [x] `at` / `setAt` accumulate into `int64_t`, not `int`

## Copy semantics
- [x] copy ctor / copy assign deep, land contiguous at offset 0
- [x] move ctor / move assign defaulted — a user copy ctor suppresses them
- [x] `clone()` — not needed, the copy constructor is it

## Validation
- [ ] assert `address.size() == ndim`
- [ ] assert `address[j] < shapes[j]`
