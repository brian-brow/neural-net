#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <SFML/Graphics.hpp>
#include <optional>
#include <vector>
#include "net/Layer.h"

// Draws the network as columns of nodes with the weight matrices as the lines
// between them. Node fill runs black (0) -> white (1); weight lines run
// red (negative) -> grey (0) -> blue (positive).
class Visualizer {
public:
  Visualizer(unsigned int width = 960, unsigned int height = 640);

  bool isOpen() const {return this->window.isOpen();}
  void pollEvents();
  void render(const Matrix& input, const std::vector<Layer>& net);

private:
  // Node centers, column 0 being the input and column i+1 being net[i].
  // Also sets zoom, which the caller uses to size nodes and labels.
  std::vector<std::vector<sf::Vector2f>> layout(const Matrix& input,
                                                const std::vector<Layer>& net);

  sf::RenderWindow window;
  std::optional<sf::Font> font;
  float nodeRadius;
  float zoom;
};

#endif
