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
