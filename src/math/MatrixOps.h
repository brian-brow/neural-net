#ifndef MATRIXOPS_H
#define MATRIXOPS_H

#include <algorithm>
#include <cmath>
#include "math/Matrix.h"

Matrix scalarMult(const Matrix& a, float s);

Matrix hadamard(const Matrix& a, const Matrix& b);

Matrix transpose(const Matrix& a);

Matrix ReLU(const Matrix& a);

Matrix ReLUPrime(const Matrix& a);

Matrix softmax(const Matrix& a);

#endif
