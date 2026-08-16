#include "math/MatrixOps.h"


Matrix scalarMult(const Matrix& a, float s)
{
  int rows = a.getRows();
  int cols = a.getCols();

  Matrix result(rows, cols);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      result.setAt(i, j, a.at(i, j) * s);
    }
  }

  return result;
}


Matrix hadamard(const Matrix& a, const Matrix& b)
{
  assert(a.getRows() == b.getRows());
  assert(a.getCols() == b.getCols());

  int rows = a.getRows();
  int cols = a.getCols();

  Matrix result(rows, cols);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      result.setAt(i, j, a.at(i, j) * b.at(i, j));
    }
  }

  return result;
}


Matrix transpose(const Matrix& a)
{
  int rows = a.getRows();
  int cols = a.getCols();

  Matrix result(cols, rows);

  for (int i = 0; i < cols; i++) {
    for (int j = 0; j < rows; j++) {
      result.setAt(i, j, a.at(j, i));
    }
  }

  return result;
}


Matrix ReLU(const Matrix& a)
{
  int rows = a.getRows();
  int cols = a.getCols();

  Matrix result(rows, cols);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      result.setAt(i, j, std::max(0.0f, a.at(i, j)));
    }
  }

  return result;
}


Matrix ReLUPrime(const Matrix& a)
{
  int rows = a.getRows();
  int cols = a.getCols();

  Matrix result(rows, cols);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (a.at(i, j) > 0) {
        result.setAt(i, j, 1);
      } else {
        result.setAt(i, j, 0);
      }
    }
  }

  return result;
}
