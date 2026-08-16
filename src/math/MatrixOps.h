#ifndef MATRIXOPS_H
#define MATRIXOPS_H

#include <algorithm>
#include "math/Matrix.h"

Matrix scalarMult(const Matrix& a, float s);

Matrix hadamard(const Matrix& a, const Matrix& b);

Matrix transpose(const Matrix& a);

Matrix ReLU(const Matrix& a);

Matrix ReLUPrime(const Matrix& a);

#endif
