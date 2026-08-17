#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include "data/Mnist.h"
#include "math/Matrix.h"
#include "math/MatrixOps.h"
#include "net/NetworkOps.h"
#include "net/Network.h"
#include "net/Layer.h"
#include "viz/Visualizer.h"

const bool TRAIN = false;

const int TRAIN_SIZE = 60000;
const int TEST_SIZE = 10000;
const int EPOCHS = 30;
const int BATCH_SIZE = 32;
const float LEARNING_RATE = 0.04f;

const std::string WEIGHTS_PATH = "weights.txt";


struct Metrics {
  float cost;
  float accuracy;
};


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


int trainAndSave(const MnistSet& test)
{
  MnistSet train = loadMnist("data/train-images-idx3-ubyte",
                          "data/train-labels-idx1-ubyte", TRAIN_SIZE);

  std::vector<int> order(TRAIN_SIZE);
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 gen(42);

  Network net({784, 128, 64, 10});

  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    std::shuffle(order.begin(), order.end(), gen);
    for (int k = 0; k < TRAIN_SIZE; k++) {
      int i = order[k];
      net.forward(train.images[i]);
      net.backward(train.targets[i]);

      if ((k + 1) % BATCH_SIZE == 0) {
        net.applyGradients(LEARNING_RATE, BATCH_SIZE);
        net.zero();
      }
    }
    if (!net.isFinite()) {
      std::cout << "epoch " << epoch << ": weights went non-finite, stopping.\n"
                << WEIGHTS_PATH << " still holds the last good epoch.\n";
      return 1;
    }

    net.save(WEIGHTS_PATH);
    Metrics m = evaluate(net, test);
    std::cout << "epoch " << epoch
              << "   cost " << m.cost
              << "   acc " << m.accuracy << "%" << std::endl;
  }

  return 0;
}


int showOneDigit(const MnistSet& test)
{
  Network net = Network::load(WEIGHTS_PATH);

  Metrics m = evaluate(net, test);
  std::cout << "loaded " << WEIGHTS_PATH
            << "   cost " << m.cost
            << "   acc " << m.accuracy << "%\n";

  std::mt19937 gen(std::random_device{}());
  std::uniform_int_distribution<int> pick(0, test.size() - 1);
  int i = pick(gen);

  net.forward(test.images[i]);
  const Matrix& out = net.getLayers().back().getOutput();
  int predicted = argmax(out);

  std::cout << "test[" << i << "]   actual " << test.labels[i]
            << "   predicted " << predicted
            << "   confidence " << (100.0f * out.at(predicted, 0)) << "%"
            << (predicted == test.labels[i] ? "" : "   <-- wrong") << '\n';

  Visualizer viz;

  while (viz.isOpen()) {
    viz.pollEvents();
    viz.render(test.images[i], net.getLayers(), test.targets[i]);
  }

  return 0;
}


int main()
{
  MnistSet test = loadMnist("data/t10k-images-idx3-ubyte",
                          "data/t10k-labels-idx1-ubyte", TEST_SIZE);

  if (TRAIN) {
    return trainAndSave(test);
  }

  try {
    return showOneDigit(test);
  } catch (const std::exception& e) {
    std::cout << e.what() << "\nRun once with TRAIN = true to create it.\n";
    return 1;
  }
}
