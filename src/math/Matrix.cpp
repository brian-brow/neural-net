#include "math/Matrix.h"
#include <random>
#include <cmath>


Matrix::Matrix(int rows, int cols)
{
  this->rows = rows;
  this->cols = cols;

  for (int i = 0; i < rows * cols; i++) {
    this->W.push_back(0);
  }
}


// Values are read in row-major order: {a, b, c, d} for a 2x2 gives [a b / c d].
Matrix::Matrix(int rows, int cols, std::initializer_list<float> values)
  : rows(rows), cols(cols), W(values)
{
  assert((int)values.size() == rows * cols);
}


float Matrix::at(int i, int j) const
{
  return W[i * cols + j];
}


void Matrix::setAt(int i, int j, float w)
{
  W.at(i * cols + j) = w;
}


void Matrix::print(std::ostream& os) const
{
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      os << W[i * cols + j]<< ' ';
    }
    os << '\n';
  }
  os << '\n';
}


void Matrix::randomize()
{
  // He initialization: normal(0, sqrt(2 / fan_in)), the standard choice for ReLU.
  // For an (out x in) weight matrix, fan_in is cols.
  static std::mt19937 gen(std::random_device{}());
  std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / cols));

  for (int i = 0; i < rows * cols; i++) {
    this->W[i] = dist(gen);
  }
}


Matrix Matrix::operator*(const Matrix& other) const
{
  assert(cols == other.rows);

  Matrix result(this->rows, other.cols);

  for (int i = 0; i < this->rows; i++) {
    for (int j = 0; j < other.cols; j++) {
      float sum = 0;
      for (int k = 0; k < other.rows; k++) {
        sum += at(i,k) * other.at(k,j);
      }
      result.setAt(i, j, sum);
    }
  }

  return result;
}


Matrix Matrix::operator+(const Matrix& other) const
{
  assert(rows == other.rows);
  assert(cols == other.cols);

  Matrix result(this->rows, other.cols);

  for (int i = 0; i < this->rows * this->cols; i++) {
    result.W[i] = this->W[i] + other.W[i];
  }

  return result;
}


Matrix Matrix::operator-(const Matrix& other) const
{
  assert(rows == other.rows);
  assert(cols == other.cols);

  Matrix result(this->rows, other.cols);

  for (int i = 0; i < this->rows * this->cols; i++) {
    result.W[i] = this->W[i] - other.W[i];
  }

  return result;
}
