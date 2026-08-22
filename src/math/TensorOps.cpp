#include "TensorOps.h"
#include <algorithm>
#include <cmath>


Tensor TensorOps::matmul(const Tensor& a, const Tensor& b)
{
  assert(a.ndim >= 1 && b.ndim >= 1);

  // a 1D operand has no matrix to offer, so invent one: on the left it becomes
  // a row (1,K), on the right a column (K,1). The invented dim is sliced back
  // off the result, which is what makes (K)x(K) come out as a 1-element dot
  // product rather than a 1x1 matrix.
  if (a.ndim == 1 || b.ndim == 1) {
    std::array<int, Tensor::MAX_DIMS> ash = a.shapes, bsh = b.shapes;
    std::array<int64_t, Tensor::MAX_DIMS> ast = a.strides, bst = b.strides;
    int aRank = a.ndim, bRank = b.ndim;

    if (a.ndim == 1) {
      ash[0] = 1;
      ash[1] = a.shapes[0];
      ast[0] = a.strides[0] * a.shapes[0];
      ast[1] = a.strides[0];
      aRank = 2;
    }

    if (b.ndim == 1) {
      bsh[0] = b.shapes[0];
      bsh[1] = 1;
      bst[0] = b.strides[0];
      bst[1] = 1;
      bRank = 2;
    }

    Tensor pa(a.data, ash, ast, aRank, a.offset);
    Tensor pb(b.data, bsh, bst, bRank, b.offset);

    Tensor r = matmul(pa, pb);

    if (b.ndim == 1 && r.ndim > 1) r = r.slice(r.ndim - 1, 0);
    if (a.ndim == 1 && r.ndim > 1) r = r.slice(r.ndim - 2, 0);

    return r;
  }

  const int M = a.shapes[a.ndim - 2];
  const int K = a.shapes[a.ndim - 1];
  const int N = b.shapes[b.ndim - 1];

  assert(b.shapes[b.ndim - 2] == K);

  // batch ranks can differ -- align them to the right and pad the shorter one
  // with implicit 1s on the left, same rule numpy uses
  const int aB = a.ndim - 2;
  const int bB = b.ndim - 2;
  const int oB = (aB > bB) ? aB : bB;

  assert(oB + 2 <= Tensor::MAX_DIMS);

  std::vector<int> outDims(oB + 2);

  // batch strides re-indexed against the output's batch rank; a dim that is
  // broadcast gets stride 0, so the odometer reads the same matrix every pass
  std::array<int64_t, Tensor::MAX_DIMS> aBatch = {};
  std::array<int64_t, Tensor::MAX_DIMS> bBatch = {};

  for (int d = 0; d < oB; d++) {
    const int ad = d - (oB - aB);
    const int bd = d - (oB - bB);

    const int as = (ad >= 0) ? a.shapes[ad] : 1;
    const int bs = (bd >= 0) ? b.shapes[bd] : 1;

    assert(as == bs || as == 1 || bs == 1);

    outDims[d] = (as > bs) ? as : bs;
    aBatch[d]  = (ad >= 0 && as != 1) ? a.strides[ad] : 0;
    bBatch[d]  = (bd >= 0 && bs != 1) ? b.strides[bd] : 0;
  }

  outDims[oB]     = M;
  outDims[oB + 1] = N;

  Tensor out(outDims);

  int64_t batch = 1;
  for (int d = 0; d < oB; d++) {
    batch *= outDims[d];
  }

  // strides inside a single matrix -- constant across the whole batch, so
  // they get hoisted out of the walk entirely
  const int64_t as0 = a.strides[a.ndim - 2],   as1 = a.strides[a.ndim - 1];
  const int64_t bs0 = b.strides[b.ndim - 2],   bs1 = b.strides[b.ndim - 1];
  const int64_t os0 = out.strides[oB],         os1 = out.strides[oB + 1];

  const float* A = a.data->data();
  const float* B = b.data->data();
  float* O = out.data->data();

  for (int64_t n = 0; n < batch; n++) {
    // decompose n into batch coords and walk each tensor's own strides to
    // the top-left of its matrix -- same odometer as Tensor::addressOf
    int64_t rem  = n;
    int64_t aOff = a.offset;
    int64_t bOff = b.offset;
    int64_t oOff = out.offset;

    for (int d = oB - 1; d >= 0; d--) {
      int64_t i = rem % outDims[d];
      rem /= outDims[d];

      aOff += i * aBatch[d];
      bOff += i * bBatch[d];
      oOff += i * out.strides[d];
    }

    // ikj rather than ijk: with j innermost both B and O are walked along
    // their last dim, and A[i][k] is loop-invariant so it lifts out entirely
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        O[oOff + i*os0 + j*os1] = 0.0f;
      }

      for (int k = 0; k < K; k++) {
        const float aik = A[aOff + i*as0 + k*as1];

        for (int j = 0; j < N; j++) {
          O[oOff + i*os0 + j*os1] += aik * B[bOff + k*bs0 + j*bs1];
        }
      }
    }
  }

  return out;
}


Tensor TensorOps::ReLU(const Tensor& t)
{
  Tensor out(t.getDims());

  for (int64_t n = 0; n < t.numel; n++) {
    (*out.data)[out.addressOf(n)] = std::max(0.0f, (*t.data)[t.addressOf(n)]);
  }

  return out;
}


Tensor TensorOps::ReLUPrime(const Tensor& t)
{
  Tensor out(t.getDims());

  for (int64_t n = 0; n < t.numel; n++) {
    (*out.data)[out.addressOf(n)] = ((*t.data)[t.addressOf(n)] > 0.0f) ? 1.0f : 0.0f;
  }

  return out;
}


Tensor TensorOps::softmax(const Tensor& t, int dim)
{
  assert(t.numel > 0);
  assert(dim >= 0 && dim < t.ndim);

  Tensor out(t.getDims());

  const int len = t.shapes[dim];
  const int64_t lanes = t.numel / len;

  for (int64_t l = 0; l < lanes; l++) {
    int64_t rem = l;
    int64_t tOff = t.offset;
    int64_t oOff = out.offset;

    for (int d = t.ndim - 1; d >= 0; d--) {
      if (d == dim) continue;
      int64_t i = rem % t.shapes[d];
      rem /= t.shapes[d];
      tOff += i * t.strides[d];
      oOff += i * out.strides[d];
    }

    float max = (*t.data)[tOff];
    for (int k = 1; k < len; k++) {
      max = std::max(max, (*t.data)[tOff + k * t.strides[dim]]);
    }

    // shifting by the max keeps exp() from overflowing; it cancels in the ratio
    float sum = 0.0f;
    for (int k = 0; k < len; k++) {
      float e = std::exp((*t.data)[tOff + k * t.strides[dim]] - max);
      (*out.data)[oOff + k * out.strides[dim]] = e;
      sum += e;
    }

    for (int k = 0; k < len; k++) {
      (*out.data)[oOff + k * out.strides[dim]] /= sum;
    }
  }

  return out;
}


// Activations are column vectors with batch dims on the left, which matmul's
// convention already forces -- so the classes are always at ndim-2.
Tensor TensorOps::softmax(const Tensor& t)
{
  assert(t.ndim >= 2);
  assert(t.shapes[t.ndim - 1] == 1);

  return softmax(t, t.ndim - 2);
}
