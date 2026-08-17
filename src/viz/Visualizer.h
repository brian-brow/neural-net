#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <SFML/Graphics.hpp>
#include <optional>
#include <vector>
#include "net/Layer.h"

// Draws the network as columns of nodes with the weight matrices as the lines
// between them. Node fill runs black (0) -> white (1); weight lines run
// red (negative) -> grey (0) -> blue (positive).
//
// Columns taller than 16 are truncated to the first and last 8 rows with a
// vertical ellipsis between, so a 784-input layer stays readable.
class Visualizer {
public:
  Visualizer(unsigned int width = 960, unsigned int height = 640);

  bool isOpen() const {return this->window.isOpen();}
  void pollEvents();
  // target is drawn as a detached reference column on the right, unconnected.
  void render(const Matrix& input, const std::vector<Layer>& net, const Matrix& target);

private:
  // A drawn node and the row it stands for. In a truncated column those differ,
  // so every lookup into a value or weight matrix must go through index.
  struct Slot {
    sf::Vector2f pos;
    int index;
  };

  struct Column {
    std::vector<Slot> slots;
    sf::Vector2f ellipsis;   // only meaningful when truncated
    int total = 0;           // rows in the real matrix
    bool truncated = false;
    bool isTarget = false;   // the reference column, not part of the network
  };

  // The input drawn as a picture, left of the input column. Only used when the
  // input is a perfect square, so 784 becomes 28x28.
  struct InputImage {
    sf::Vector2f topLeft;
    float cell = 0.0f;
    int side = 0;
    bool visible = false;
  };

  // Also sets zoom and inputImage, which render() uses.
  std::vector<Column> layout(const Matrix& input, const std::vector<Layer>& net,
                             const Matrix& target);
  void drawEllipsis(const Column& column);
  void drawInputImage(const Matrix& input);

  sf::RenderWindow window;
  std::optional<sf::Font> font;
  float nodeRadius;
  float zoom;
  InputImage inputImage;
};

#endif
