#ifndef NETWORKOPS_H
#define NETWORKOPS_H

#include <cassert>
#include <cmath>
#include "math/Matrix.h"
#include "math/MatrixOps.h"

float mse(const Matrix& a, const Matrix& y);

Matrix msePrime(const Matrix& a, const Matrix& y);

float crossEntropy(const Matrix& a, const Matrix& y);

Matrix crossEntropyPrime(const Matrix& a, const Matrix& y);

#endif
