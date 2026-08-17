#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <stdlib.h>
#include <cassert>
#include <iostream>

class Matrix {
public:
  Matrix(int rows, int cols);
  Matrix(int rows, int cols, std::initializer_list<float> values);
  // Matrix(Matrix &&) = default;
  // Matrix(const Matrix &) = default;

  float at(int i, int j) const;
  void setAt(int i, int j, float w);

  int getRows() const {return this->rows;}
  int getCols() const {return this->cols;}

  void print(std::ostream& os = std::cout) const;
  void randomize();
  void zero();

  Matrix operator+(const Matrix& other) const;
  Matrix operator-(const Matrix& other) const;
  Matrix operator*(const Matrix& other) const;

private:
  int rows, cols;
  std::vector<float> W;
};

#endif
