#ifndef METRICS_H
#define METRICS_H

#include "data/Mnist.h"
#include "math/Matrix.h"
#include "net/Network.h"

struct Metrics {
  float cost;
  float accuracy;
};

int argmax(const Matrix& m);

Metrics evaluate(Network& net, const MnistSet& set);

#endif
