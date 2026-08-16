#include "viz/Visualizer.h"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

// Activations at or above this value render as pure white. ReLU output is
// unbounded, so raise this if everything looks blown out.
const float ACTIVATION_MAX = 1.0f;

const float MARGIN           = 90.0f;
const float BASE_RADIUS      = 22.0f;
const float BASE_SPACING     = 95.0f;   // vertical, between nodes
const float BASE_COL_SPACING = 190.0f;  // horizontal, between layers
const float MAX_ZOOM         = 2.0f;    // how far the graph may scale up
const float MIN_THICK        = 2.0f;
const float MAX_THICK        = 4.5f;

const sf::Color BACKGROUND(24, 24, 30);
const sf::Color NEUTRAL(122, 122, 134);   // a weight of 0
const sf::Color NEGATIVE(235, 62, 52);    // most negative weight
const sf::Color POSITIVE(58, 118, 240);   // most positive weight

// A line must stay at least this far above the background in luminance.
const float MIN_CONTRAST = 55.0f;

std::uint8_t lerp(std::uint8_t a, std::uint8_t b, float t)
{
  return static_cast<std::uint8_t>(a + (b - a) * t);
}

sf::Color mix(sf::Color a, sf::Color b, float t)
{
  return sf::Color(lerp(a.r, b.r, t), lerp(a.g, b.g, t), lerp(a.b, b.b, t));
}

float luminance(sf::Color c)
{
  return 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;
}

// Structural guarantee that a weight line can never sink into the background:
// if it is not bright enough, walk it toward white until it clears the floor.
sf::Color ensureVisible(sf::Color c)
{
  float deficit = MIN_CONTRAST - (luminance(c) - luminance(BACKGROUND));
  if (deficit <= 0.0f) {
    return c;
  }

  float headroom = 255.0f - luminance(c);
  float t = headroom > 0.0f ? std::min(1.0f, deficit / headroom) : 1.0f;

  return mix(c, sf::Color::White, t);
}

sf::Color activationColor(float a)
{
  float t = std::clamp(a / ACTIVATION_MAX, 0.0f, 1.0f);
  auto v = static_cast<std::uint8_t>(t * 255.0f);
  return sf::Color(v, v, v);
}

// scale is the largest |weight| in the net, so the extremes always saturate
// no matter how the weights are initialized.
sf::Color weightColor(float w, float scale)
{
  if (scale <= 0.0f) {
    return NEUTRAL;
  }

  float t = std::clamp(w / scale, -1.0f, 1.0f);
  sf::Color c = t >= 0.0f ? mix(NEUTRAL, POSITIVE, t) : mix(NEUTRAL, NEGATIVE, -t);

  return ensureVisible(c);
}

float maxAbsWeight(const std::vector<Layer>& net)
{
  float scale = 0.0f;

  for (const Layer& l : net) {
    Matrix W = l.getWeights();
    for (int i = 0; i < W.getRows(); i++) {
      for (int j = 0; j < W.getCols(); j++) {
        scale = std::max(scale, std::abs(W.at(i, j)));
      }
    }
  }

  return scale;
}

// SFML has no thick-line primitive, so each connection is a rectangle rotated
// onto the segment.
void drawLine(sf::RenderWindow& window, sf::Vector2f a, sf::Vector2f b,
              sf::Color color, float thickness)
{
  sf::Vector2f d = b - a;
  float length = std::sqrt(d.x * d.x + d.y * d.y);

  sf::RectangleShape line({length, thickness});
  line.setOrigin({0.0f, thickness / 2.0f});
  line.setPosition(a);
  line.setRotation(sf::radians(std::atan2(d.y, d.x)));
  line.setFillColor(color);

  window.draw(line);
}

std::optional<sf::Font> loadFont()
{
  const char* candidates[] = {
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
  };

  for (const char* path : candidates) {
    sf::Font f;
    if (f.openFromFile(path)) {
      return f;
    }
  }

  return std::nullopt;  // labels are skipped, the diagram still draws
}

}  // namespace


namespace {

// Thin rotated rectangles drop out badly without multisampling, which is the
// other reason weak weights looked like they vanished.
sf::ContextSettings antialiased()
{
  sf::ContextSettings settings;
  settings.antiAliasingLevel = 8;
  return settings;
}

}  // namespace


Visualizer::Visualizer(unsigned int width, unsigned int height)
  : window(sf::VideoMode({width, height}), "Neural Net",
           sf::Style::Default, sf::State::Windowed, antialiased()),
    font(loadFont()),
    nodeRadius(BASE_RADIUS),
    zoom(1.0f)
{
  window.setFramerateLimit(60);
}


void Visualizer::pollEvents()
{
  while (const std::optional event = window.pollEvent()) {
    if (event->is<sf::Event::Closed>()) {
      window.close();
    }

    // Without this the view keeps the original size while layout() works in
    // the new one, so everything lands off screen.
    if (const auto* resized = event->getIf<sf::Event::Resized>()) {
      window.setView(sf::View(sf::FloatRect({0.0f, 0.0f}, sf::Vector2f(resized->size))));
    }

    if (const auto* key = event->getIf<sf::Event::KeyPressed>()) {
      if (key->code == sf::Keyboard::Key::Escape || key->code == sf::Keyboard::Key::Q) {
        window.close();
      }
    }
  }
}


std::vector<std::vector<sf::Vector2f>> Visualizer::layout(
    const Matrix& input, const std::vector<Layer>& net)
{
  std::vector<int> counts;
  counts.push_back(input.getRows());
  for (const Layer& l : net) {
    counts.push_back(l.getOutput().getRows());
  }

  auto size = window.getSize();
  float usableW = size.x - 2 * MARGIN;
  float usableH = size.y - 2 * MARGIN;
  int columns = static_cast<int>(counts.size());
  int tallest = *std::max_element(counts.begin(), counts.end());

  // One uniform zoom for both axes, so the graph grows into a fullscreen window
  // and shrinks to fit a small one without ever being stretched out of shape.
  float fitW = columns > 1 ? usableW / ((columns - 1) * BASE_COL_SPACING) : MAX_ZOOM;
  float fitH = tallest > 1 ? usableH / ((tallest - 1) * BASE_SPACING) : MAX_ZOOM;
  zoom = std::min({fitW, fitH, MAX_ZOOM});

  float colSpacing = BASE_COL_SPACING * zoom;
  float spacing    = BASE_SPACING * zoom;
  float left = size.x / 2.0f - colSpacing * (columns - 1) / 2.0f;

  std::vector<std::vector<sf::Vector2f>> positions;

  for (int c = 0; c < columns; c++) {
    float x = left + colSpacing * c;

    int n = counts[c];
    float top = size.y / 2.0f - spacing * (n - 1) / 2.0f;

    std::vector<sf::Vector2f> column;
    for (int i = 0; i < n; i++) {
      column.push_back({x, top + spacing * i});
    }

    positions.push_back(column);
  }

  return positions;
}


void Visualizer::render(const Matrix& input, const std::vector<Layer>& net)
{
  window.clear(BACKGROUND);

  auto positions = layout(input, net);
  float scale = maxAbsWeight(net);

  // Nodes track the zoom, but never grow into their neighbours.
  nodeRadius = BASE_RADIUS * zoom;
  for (const auto& column : positions) {
    if (column.size() > 1) {
      float gap = column[1].y - column[0].y;
      nodeRadius = std::min(nodeRadius, gap * 0.38f);
    }
  }

  // Connections first so the nodes sit on top of them.
  for (size_t l = 0; l < net.size(); l++) {
    Matrix W = net[l].getWeights();          // (out x in)
    const auto& from = positions[l];
    const auto& to   = positions[l + 1];

    for (int j = 0; j < W.getRows(); j++) {
      for (int k = 0; k < W.getCols(); k++) {
        float w = W.at(j, k);
        float strength = scale > 0.0f ? std::abs(w) / scale : 0.0f;
        float thickness = MIN_THICK + (MAX_THICK - MIN_THICK) * strength;

        drawLine(window, from[k], to[j], weightColor(w, scale), thickness);
      }
    }
  }

  // Column 0 is the raw input, the rest are layer activations.
  for (size_t c = 0; c < positions.size(); c++) {
    Matrix values = c == 0 ? input : net[c - 1].getOutput();

    for (size_t i = 0; i < positions[c].size(); i++) {
      sf::CircleShape node(nodeRadius);
      node.setOrigin({nodeRadius, nodeRadius});
      node.setPosition(positions[c][i]);
      node.setFillColor(activationColor(values.at(static_cast<int>(i), 0)));
      node.setOutlineThickness(1.5f);
      node.setOutlineColor(sf::Color(90, 90, 100));

      window.draw(node);
    }
  }

  if (font) {
    for (size_t c = 0; c < positions.size(); c++) {
      auto size = static_cast<unsigned int>(16 * zoom);
      sf::Text label(*font, c == 0 ? "x" : "L" + std::to_string(c), size);
      label.setFillColor(sf::Color(150, 150, 160));

      auto bounds = label.getLocalBounds();
      label.setPosition({positions[c][0].x - bounds.size.x / 2.0f,
                         positions[c][0].y - nodeRadius - 34.0f});

      window.draw(label);
    }
  }

  window.display();
}
