#include "app/Visualize.h"

#include <iostream>
#include <random>
#include "app/Config.h"
#include "app/Metrics.h"
#include "net/Network.h"
#include "viz/Visualizer.h"


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
