#ifndef METRICS_H
#define METRICS_H

#include "data/Mnist.h"
#include "math/Matrix.h"
#include "math/Tensor.h"
#include "net/Network.h"

struct Metrics {
  float cost;
  float accuracy;
};

int argmax(const Matrix& m);

int argmax(const Tensor& t);

Metrics evaluate(Network& net, const MnistSet& set);

#endif
