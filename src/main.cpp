#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <vector>
#include "math/Matrix.h"
#include "math/MatrixOps.h"
#include "net/Layer.h"
#include "viz/Visualizer.h"

// Open an SFML window showing the network instead of only printing it.
const bool VISUALIZE = true;


int main()
{
  srand(time(NULL));

  // 5-wide input feeding 4 layers of 5 nodes each.
  // Each Layer He-randomizes its own W and starts with b = 0.
  const int WIDTH  = 5;
  const int LAYERS = 4;

  Matrix x(WIDTH, 1);
  x.randomize();

  std::vector<Layer> net;
  for (int i = 0; i < LAYERS; i++) {
    net.emplace_back(WIDTH, WIDTH);
  }

  net[0].forward(x);
  for (int i = 1; i < LAYERS; i++) {
    net[i].forward(net[i - 1].getOutput());
  }

  std::cout << "x\n";
  x.print();

  for (int i = 0; i < LAYERS; i++) {
    std::cout << "z" << i + 1 << '\n';
    net[i].getPreAct().print();

    std::cout << "a" << i + 1 << '\n';
    net[i].getOutput().print();
  }

  if (VISUALIZE) {
    Visualizer viz;

    while (viz.isOpen()) {
      viz.pollEvents();
      viz.render(x, net);
    }
  }

  return 0;
}
