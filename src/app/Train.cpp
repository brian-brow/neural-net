#include "app/Train.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include "app/Config.h"
#include "app/Metrics.h"
#include "net/Network.h"


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
