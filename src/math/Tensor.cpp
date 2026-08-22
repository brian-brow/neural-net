#include "Tensor.h"


Tensor::Tensor(const std::vector<int>& d)
{
  int len = d.size();

  assert(len > 0);
  assert(len <= MAX_DIMS);

  this->offset = 0;
  this->ndim = len;

  int64_t tmp = 1;
  for (int i = 0; i < ndim; i++) {
    this->shapes[i] = d[i];
    tmp *= shapes[i];
  }

  this->numel = tmp;

  int64_t stride = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shapes[i];
  }

  this->data = std::make_shared<std::vector<float>>(numel, 0.0f);

  computeContiguous();
}


Tensor::Tensor(std::shared_ptr<std::vector<float>> buf,
               const std::array<int, MAX_DIMS>& sh,
               const std::array<int64_t, MAX_DIMS>& st,
               int nd,
               int64_t off)
  : data(std::move(buf)), offset(off), ndim(nd), shapes(sh), strides(st)
{
  int64_t tmp = 1;
  for (int i = 0; i < ndim; i++) {
    tmp *= shapes[i];
  }
  this->numel = tmp;

  computeContiguous();
}


// The copy takes other's shape but not its strides -- a copy of a transposed
// view is a plain row-major tensor of the transposed shape, not another view.
Tensor::Tensor(const Tensor& other)
  : data(std::make_shared<std::vector<float>>(other.numel)),
    offset(0),
    ndim(other.ndim),
    numel(other.numel),
    contiguous(true),
    shapes(other.shapes)
{
  int64_t stride = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shapes[i];
  }

  for (int64_t n = 0; n < numel; n++) {
    (*data)[n] = (*other.data)[other.addressOf(n)];
  }
}


Tensor& Tensor::operator=(const Tensor& other)
{
  if (this != &other) {
    *this = Tensor(other);
  }
  return *this;
}


float Tensor::at(const std::vector<int>& address) const
{
  int64_t i = offset;
  for (int j = 0; j < ndim; j++) {
    i += strides[j] * address[j];
  }
  return data->at(i);
}


void Tensor::setAt(const std::vector<int>& address, float w)
{
  int64_t i = offset;
  for (int j = 0; j < ndim; j++) {
    i += strides[j] * address[j];
  }
  data->at(i) = w;
}


Tensor Tensor::slice(int dim, int index) const
{
  assert(ndim > 1);
  assert(dim >= 0 && dim < ndim);
  assert(index >= 0 && index < shapes[dim]);

  std::array<int, MAX_DIMS> sh = {};
  std::array<int64_t, MAX_DIMS> st = {};

  for (int i = 0, j = 0; i < ndim; i++) {
    if (i == dim) continue;      // drop this one
    sh[j] = shapes[i];
    st[j] = strides[i];
    j++;                         // j only advances when we keep
  }

  return Tensor(data, sh, st, ndim - 1,
                offset + (int64_t)index * strides[dim]);
}


Tensor Tensor::transpose() const
{
  assert(ndim > 1);

  std::array<int, MAX_DIMS> sh = shapes;
  sh[ndim-2] = shapes[ndim-1];
  sh[ndim-1] = shapes[ndim-2];

  std::array<int64_t, MAX_DIMS> st = strides;
  st[ndim-2] = strides[ndim-1];
  st[ndim-1] = strides[ndim-2];

  return Tensor(data, sh, st, ndim, offset);
}


void Tensor::randomize()
{
  assert(ndim >= 2);

  int64_t fanIn = 1;
  for (int i = 1; i < ndim; i++) {
    fanIn *= shapes[i];
  }

  static std::mt19937 gen(std::random_device{}());
  std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / fanIn));

  for (int64_t n = 0; n < numel; n++) {
    (*data)[addressOf(n)] = dist(gen);
  }
}


void Tensor::zero()
{
  for (int64_t n = 0; n < numel; n++) {
    (*data)[addressOf(n)] = 0.0f;
  }
}


Tensor Tensor::operator+(const Tensor& other) const
{
  assert(ndim == other.ndim);
  assert(shapes == other.shapes);

  Tensor result(getDims());

  for (int64_t n = 0; n < numel; n++) {
    (*result.data)[result.addressOf(n)] =
      (*data)[addressOf(n)] + (*other.data)[other.addressOf(n)];
  }

  return result;
}


Tensor Tensor::operator-(const Tensor& other) const
{
  assert(ndim == other.ndim);
  assert(shapes == other.shapes);

  Tensor result(getDims());

  for (int64_t n = 0; n < numel; n++) {
    (*result.data)[result.addressOf(n)] =
      (*data)[addressOf(n)] - (*other.data)[other.addressOf(n)];
  }

  return result;
}


Tensor Tensor::operator*(const Tensor& other) const
{
  assert(ndim == other.ndim);
  assert(shapes == other.shapes);

  Tensor result(getDims());

  for (int64_t n = 0; n < numel; n++) {
    (*result.data)[result.addressOf(n)] =
      (*data)[addressOf(n)] * (*other.data)[other.addressOf(n)];
  }

  return result;
}


Tensor Tensor::operator*(float s) const
{
  Tensor result(getDims());

  for (int64_t n = 0; n < numel; n++) {
    (*result.data)[result.addressOf(n)] = (*data)[addressOf(n)] * s;
  }

  return result;
}


void Tensor::print(std::ostream& os) const
{
  for (int64_t n = 0; n < numel; n++) {
    // count how many of the innermost dims just rolled back to 0 -- that is
    // how many line breaks this element needs in front of it
    int nl = 0;
    int64_t rem = n;

    for (int d = ndim - 1; d >= 0; d--) {
      if (rem % shapes[d] != 0) {
        break;
      }
      rem /= shapes[d];
      nl++;
    }

    if (n > 0) {
      for (int k = 0; k < nl; k++) {
        os << '\n';
      }
    }

    os << (*data)[addressOf(n)] << ' ';
  }
  os << '\n';
}


// Contiguous means the strides are exactly what the row-major loop in the
// first constructor would generate for this shape.
void Tensor::computeContiguous()
{
  int64_t expected = 1;

  for (int d = ndim - 1; d >= 0; d--) {
    if (shapes[d] != 1 && strides[d] != expected) {
      this->contiguous = false;
      return;
    }
    expected *= shapes[d];
  }

  this->contiguous = true;
}


// Flat element number (0..numel-1, last dim fastest) to an index into data.
// Contiguous is the common case and skips the decomposition entirely; note it
// still has to start at offset, since a contiguous view can begin partway into
// the buffer.
int64_t Tensor::addressOf(int64_t n) const
{
  if (contiguous) {
    return offset + n;
  }

  int64_t rem = n;
  int64_t i = offset;

  for (int d = ndim - 1; d >= 0; d--) {
    i += strides[d] * (rem % shapes[d]);
    rem /= shapes[d];
  }

  return i;
}
