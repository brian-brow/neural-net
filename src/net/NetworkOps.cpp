#include "net/NetworkOps.h"


float mse(const Matrix& a, const Matrix& y)
{
  int rows = a.getRows();

  assert(rows == y.getRows());
  assert(a.getCols() == y.getCols());
  assert(a.getCols() == 1);

  int n = rows;
  float sum = 0.0f;

  for (int i = 0; i < n; i++) {
    float err = (a.at(i, 0) - y.at(i, 0));
    sum += err * err;
  }

  return sum / n;
}


Matrix msePrime(const Matrix& a, const Matrix& y)
{
  int rows = a.getRows();

  assert(rows == y.getRows());
  assert(a.getCols() == y.getCols());
  assert(a.getCols() == 1);

  return scalarMult((a - y), (2.0f / rows));
}


float crossEntropy(const Matrix& a, const Matrix& y)
{
  int rows = a.getRows();

  assert(rows == y.getRows());
  assert(a.getCols() == y.getCols());
  assert(a.getCols() == 1);

  float sum = 0.0f;

  for (int i = 0; i < rows; i++) {
    float p = std::max(a.at(i, 0), 1e-7f);
    sum += y.at(i, 0) * std::log(p);
  }

  return -sum;
}


Matrix crossEntropyPrime(const Matrix& a, const Matrix& y)
{
  assert(a.getRows() == y.getRows());
  assert(a.getCols() == y.getCols());
  assert(a.getCols() == 1);

  return a - y;
}


float mse(const Tensor& a, const Tensor& y)
{
  std::vector<int> d = a.getDims();

  assert(d == y.getDims());
  assert(d.size() == 2);
  assert(d[1] == 1);

  int n = d[0];
  float sum = 0.0f;

  for (int i = 0; i < n; i++) {
    float err = a.at({i, 0}) - y.at({i, 0});
    sum += err * err;
  }

  return sum / n;
}


Tensor msePrime(const Tensor& a, const Tensor& y)
{
  std::vector<int> d = a.getDims();

  assert(d == y.getDims());
  assert(d.size() == 2);
  assert(d[1] == 1);

  return (a - y) * (2.0f / d[0]);
}


float crossEntropy(const Tensor& a, const Tensor& y)
{
  std::vector<int> d = a.getDims();

  assert(d == y.getDims());
  assert(d.size() == 2);
  assert(d[1] == 1);

  float sum = 0.0f;

  for (int i = 0; i < d[0]; i++) {
    float p = std::max(a.at({i, 0}), 1e-7f);
    sum += y.at({i, 0}) * std::log(p);
  }

  return -sum;
}


Tensor crossEntropyPrime(const Tensor& a, const Tensor& y)
{
  std::vector<int> d = a.getDims();

  assert(d == y.getDims());
  assert(d.size() == 2);
  assert(d[1] == 1);

  return a - y;
}
