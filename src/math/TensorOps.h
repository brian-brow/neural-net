#ifndef TENSOROPS_H
#define TENSOROPS_H

#include "Tensor.h"

// A struct of statics rather than free functions so Tensor needs a single
// friend declaration to cover every op, present and future. Ops go here rather
// than on Tensor when they read the raw buffer in a hot loop -- at() builds a
// std::vector per element access, which is fine for tests and fatal in matmul.
struct TensorOps {

  // Last two dims are the matrix, everything left of them is batch. One walk
  // over the flattened batch count covers every rank -- 2D is just the case
  // where that count comes out to 1.
  static Tensor matmul(const Tensor& a, const Tensor& b);

  static Tensor ReLU(const Tensor& t);
  static Tensor ReLUPrime(const Tensor& t);
  static Tensor softmax(const Tensor& t, int dim);
  static Tensor softmax(const Tensor& t);
};

#endif
