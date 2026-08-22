#include "app/Visualize.h"

#include <iostream>
#include <random>
#include "app/Config.h"
#include "app/Metrics.h"
#include "net/Network.h"
#include "viz/Visualizer.h"


namespace {

int randomIndex(const MnistSet& test, std::mt19937& gen)
{
  std::uniform_int_distribution<int> pick(0, test.size() - 1);
  return pick(gen);
}


// The net is left holding this sample's activations, which is what the
// visualizer draws.
void showSample(Network& net, const MnistSet& test, int i)
{
  net.forward(test.images[i]);
  const Tensor& out = net.getLayers().back().getOutput();
  int predicted = argmax(out);

  std::cout << "test[" << i << "]   actual " << test.labels[i]
            << "   predicted " << predicted
            << "   confidence " << (100.0f * out.at({predicted, 0})) << "%"
            << (predicted == test.labels[i] ? "" : "   <-- wrong") << '\n';
}

}  // namespace


int showOneDigit(const MnistSet& test)
{
  Network net = Network::load(WEIGHTS_PATH);

  Metrics m = evaluate(net, test);
  std::cout << "loaded " << WEIGHTS_PATH
            << "   cost " << m.cost
            << "   acc " << m.accuracy << "%\n";

  std::mt19937 gen(std::random_device{}());
  int i = randomIndex(test, gen);
  showSample(net, test, i);

  Visualizer viz(test.size());
  viz.setIndex(i);

  while (viz.isOpen()) {
    Visualizer::Request request = viz.pollEvents();

    std::optional<int> next;
    if (request.index) {
      next = *request.index;
    } else if (request.randomSample) {
      next = randomIndex(test, gen);
    }

    // Before render, so a new sample is on screen the same frame rather than
    // one frame stale.
    if (next) {
      i = *next;
      showSample(net, test, i);
      viz.setIndex(i);
    }

    viz.render(test.images[i], net.getLayers(), test.targets[i]);
  }

  return 0;
}
