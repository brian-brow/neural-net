#include "app/Metrics.h"

#include "net/NetworkOps.h"


// Index of the largest element in a column vector: the predicted digit.
int argmax(const Matrix& m)
{
  int best = 0;

  for (int i = 1; i < m.getRows(); i++) {
    if (m.at(i, 0) > m.at(best, 0)) {
      best = i;
    }
  }

  return best;
}


Metrics evaluate(Network& net, const MnistSet& set)
{
  float totalCost = 0.0f;
  int correct = 0;

  for (int i = 0; i < set.size(); i++) {
    net.forward(set.images[i]);
    const Matrix& out = net.getLayers().back().getOutput();

    totalCost += crossEntropy(out, set.targets[i]);

    if (argmax(out) == set.labels[i]) {
      correct++;
    }
  }

  return {totalCost / set.size(), 100.0f * correct / set.size()};
}
