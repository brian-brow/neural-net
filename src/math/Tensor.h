#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <array>
#include <stdlib.h>
#include <cassert>
#include <iostream>
#include <memory>
#include <random>

struct TensorOps;

class Tensor {
  friend struct TensorOps;

public:
  static const int MAX_DIMS = 8;

  // construction
  Tensor(const std::vector<int>& d);
  Tensor(std::shared_ptr<std::vector<float>> buf,
         const std::array<int, MAX_DIMS>& sh,
         const std::array<int64_t, MAX_DIMS>& st,
         int nd,
         int64_t off);

  // copies are deep and land contiguous; aliasing is only ever via slice /
  // transpose, which say so at the call site. The moves have to be spelled out
  // -- declaring a copy constructor suppresses the implicit ones.
  Tensor(const Tensor& other);
  Tensor& operator=(const Tensor& other);
  Tensor(Tensor&& other) = default;
  Tensor& operator=(Tensor&& other) = default;
  ~Tensor() = default;

  // element access
  float at(const std::vector<int>& address) const;
  void setAt(const std::vector<int>& address, float w);

  // metadata
  int getNDims() const {return this->ndim;}
  int64_t getNumel() const {return this->numel;}
  int64_t getOffset() const {return this->offset;}
  bool isContiguous() const {return this->contiguous;}
  std::vector<int> getDims() const {return std::vector<int>(shapes.begin(), shapes.begin() + ndim);}
  std::vector<int64_t> getStrides() const {return std::vector<int64_t>(strides.begin(), strides.begin() + ndim);}

  // views -- share the buffer, rewrite the metadata
  Tensor slice(int dim, int index) const;
  Tensor transpose() const;

  // whole-tensor writes
  void randomize();
  void zero();

  // elementwise
  Tensor operator+(const Tensor& other) const;
  Tensor operator-(const Tensor& other) const;
  Tensor operator*(const Tensor& other) const;
  Tensor operator*(float s) const;

  void print(std::ostream& os = std::cout) const;

private:
  void computeContiguous();
  int64_t addressOf(int64_t n) const;

  std::shared_ptr<std::vector<float>> data;

  int64_t offset;
  int ndim;
  int64_t numel;
  bool contiguous;

  std::array<int, MAX_DIMS> shapes = {};
  std::array<int64_t, MAX_DIMS> strides = {};
};

#endif
