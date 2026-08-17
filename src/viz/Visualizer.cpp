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

// Above this many rows a column is drawn as first HALF, ellipsis, last HALF.
const int MAX_VISIBLE = 16;
const int HALF        = MAX_VISIBLE / 2;

// Gap between the input picture and the input column, as a fraction of the
// picture's size.
const float IMAGE_GAP_RATIO = 0.18f;

// Extra spacing before the target column, in units of the normal column gap,
// so it reads as detached from the network.
const float TARGET_EXTRA_GAP = 0.55f;

// Drawn nodes, plus one blank slot for the ellipsis when truncated.
int slotCount(int rows)
{
  return rows <= MAX_VISIBLE ? rows : MAX_VISIBLE + 1;
}

// Side length if rows is a perfect square (784 -> 28), otherwise 0.
int squareSide(int rows)
{
  int r = static_cast<int>(std::sqrt(static_cast<double>(rows)));

  while (r > 0 && r * r > rows) {
    r--;
  }
  while ((r + 1) * (r + 1) <= rows) {
    r++;
  }

  return r * r == rows ? r : 0;
}

const sf::Color BACKGROUND(24, 24, 30);
const sf::Color NEUTRAL(122, 122, 134);   // a weight of 0
const sf::Color NEGATIVE(235, 62, 52);    // most negative weight
const sf::Color POSITIVE(58, 118, 240);   // most positive weight
const sf::Color TARGET_OUTLINE(196, 154, 74);  // reference column, not a layer

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


std::vector<Visualizer::Column> Visualizer::layout(
    const Matrix& input, const std::vector<Layer>& net, const Matrix& target)
{
  // Last entry is the target, which is drawn but never wired to anything.
  std::vector<int> counts;
  counts.push_back(input.getRows());
  for (const Layer& l : net) {
    counts.push_back(l.getOutput().getRows());
  }
  counts.push_back(target.getRows());

  auto size = window.getSize();
  float usableW = size.x - 2 * MARGIN;
  float usableH = size.y - 2 * MARGIN;
  int columns = static_cast<int>(counts.size());

  // The target sits a wider gap out than a normal column, so the break in the
  // chain is visible. Measured in units of colSpacing.
  float gapUnits = (columns - 1) + TARGET_EXTRA_GAP;

  // Fit against what is actually drawn, not the true row counts, or a 784-row
  // layer would squash everything to nothing.
  int tallest = 0;
  for (int n : counts) {
    tallest = std::max(tallest, slotCount(n));
  }

  // One uniform zoom for both axes, so the graph grows into a fullscreen window
  // and shrinks to fit a small one without ever being stretched out of shape.
  float fitW = gapUnits > 0.0f ? usableW / (gapUnits * BASE_COL_SPACING) : MAX_ZOOM;
  float fitH = tallest > 1 ? usableH / ((tallest - 1) * BASE_SPACING) : MAX_ZOOM;
  zoom = std::min({fitW, fitH, MAX_ZOOM});

  float colSpacing = BASE_COL_SPACING * zoom;
  float spacing    = BASE_SPACING * zoom;

  // A square input gets drawn as a picture beside its column. Size it to the
  // column's height, then clamp to whatever width the graph is not using.
  int side = squareSide(counts[0]);
  float columnHeight = spacing * (slotCount(counts[0]) - 1);
  float graphWidth = colSpacing * gapUnits;

  float imageSize = 0.0f;
  float imageGap  = 0.0f;

  if (side >= 4) {
    float room = std::max(0.0f, usableW - graphWidth);
    imageSize = std::min(columnHeight, room / (1.0f + IMAGE_GAP_RATIO));
    imageGap  = imageSize * IMAGE_GAP_RATIO;
  }

  // Centre the picture and the graph together, not the graph alone.
  float startX = size.x / 2.0f - (imageSize + imageGap + graphWidth) / 2.0f;
  float left = startX + imageSize + imageGap;

  inputImage.visible = imageSize > 1.0f;
  inputImage.side    = side;
  inputImage.cell    = side > 0 ? imageSize / side : 0.0f;
  inputImage.topLeft = {startX, size.y / 2.0f - imageSize / 2.0f};

  std::vector<Column> layout;

  for (int c = 0; c < columns; c++) {
    bool isTarget = c == columns - 1;

    // Everything is evenly spaced except the target, which is pushed out.
    float x = left + colSpacing * (isTarget ? c + TARGET_EXTRA_GAP : c);

    int n = counts[c];
    int slots = slotCount(n);
    float top = size.y / 2.0f - spacing * (slots - 1) / 2.0f;

    Column column;
    column.total = n;
    column.truncated = n > MAX_VISIBLE;
    column.isTarget = isTarget;

    if (!column.truncated) {
      for (int i = 0; i < n; i++) {
        column.slots.push_back({{x, top + spacing * i}, i});
      }
    } else {
      // First HALF rows, a gap holding the ellipsis, then the last HALF rows.
      for (int i = 0; i < HALF; i++) {
        column.slots.push_back({{x, top + spacing * i}, i});
      }

      column.ellipsis = {x, top + spacing * HALF};

      for (int i = 0; i < HALF; i++) {
        column.slots.push_back({{x, top + spacing * (HALF + 1 + i)}, n - HALF + i});
      }
    }

    layout.push_back(column);
  }

  return layout;
}


void Visualizer::drawInputImage(const Matrix& input)
{
  if (!inputImage.visible) {
    return;
  }

  float cell = inputImage.cell;
  int side = inputImage.side;

  for (int r = 0; r < side; r++) {
    for (int c = 0; c < side; c++) {
      // Slightly oversized so neighbouring cells never leave a seam.
      sf::RectangleShape pixel({cell + 0.5f, cell + 0.5f});
      pixel.setPosition({inputImage.topLeft.x + cell * c,
                         inputImage.topLeft.y + cell * r});
      pixel.setFillColor(activationColor(input.at(r * side + c, 0)));

      window.draw(pixel);
    }
  }

  sf::RectangleShape border({cell * side, cell * side});
  border.setPosition(inputImage.topLeft);
  border.setFillColor(sf::Color::Transparent);
  border.setOutlineThickness(1.5f);
  border.setOutlineColor(sf::Color(90, 90, 100));

  window.draw(border);
}


void Visualizer::drawEllipsis(const Column& column)
{
  float dot = std::max(1.5f, nodeRadius * 0.16f);
  float step = nodeRadius * 0.62f;

  for (int i = -1; i <= 1; i++) {
    sf::CircleShape mark(dot);
    mark.setOrigin({dot, dot});
    mark.setPosition({column.ellipsis.x, column.ellipsis.y + step * i});
    mark.setFillColor(sf::Color(140, 140, 150));

    window.draw(mark);
  }
}


void Visualizer::render(const Matrix& input, const std::vector<Layer>& net,
                        const Matrix& target)
{
  window.clear(BACKGROUND);

  auto columns = layout(input, net, target);
  float scale = maxAbsWeight(net);

  // One value matrix per column, in the same order layout() built them.
  std::vector<Matrix> values;
  values.push_back(input);
  for (const Layer& l : net) {
    values.push_back(l.getOutput());
  }
  values.push_back(target);

  // Nodes track the zoom, but never grow into their neighbours.
  nodeRadius = BASE_RADIUS * zoom;
  for (const auto& column : columns) {
    if (column.slots.size() > 1) {
      float gap = column.slots[1].pos.y - column.slots[0].pos.y;
      nodeRadius = std::min(nodeRadius, gap * 0.38f);
    }
  }

  // Connections first so the nodes sit on top of them. Only weights joining two
  // drawn nodes are shown, so every line on screen is a real W entry.
  for (size_t l = 0; l < net.size(); l++) {
    Matrix W = net[l].getWeights();          // (out x in)
    const auto& from = columns[l];
    const auto& to   = columns[l + 1];

    for (const Slot& out : to.slots) {
      for (const Slot& in : from.slots) {
        float w = W.at(out.index, in.index);
        float strength = scale > 0.0f ? std::abs(w) / scale : 0.0f;
        float thickness = MIN_THICK + (MAX_THICK - MIN_THICK) * strength;

        drawLine(window, in.pos, out.pos, weightColor(w, scale), thickness);
      }
    }
  }

  for (size_t c = 0; c < columns.size(); c++) {
    // The reference column gets a warmer outline so it is not mistaken for a layer.
    sf::Color outline = columns[c].isTarget ? TARGET_OUTLINE : sf::Color(90, 90, 100);

    for (const Slot& slot : columns[c].slots) {
      sf::CircleShape node(nodeRadius);
      node.setOrigin({nodeRadius, nodeRadius});
      node.setPosition(slot.pos);
      node.setFillColor(activationColor(values[c].at(slot.index, 0)));
      node.setOutlineThickness(1.5f);
      node.setOutlineColor(outline);

      window.draw(node);
    }

    if (columns[c].truncated) {
      drawEllipsis(columns[c]);
    }
  }

  drawInputImage(input);

  if (font) {
    for (size_t c = 0; c < columns.size(); c++) {
      // Truncated columns carry their real size, so the hidden rows are obvious.
      std::string name = columns[c].isTarget ? "y" :
                         c == 0 ? "x" : "L" + std::to_string(c);
      if (columns[c].truncated) {
        name += " (" + std::to_string(columns[c].total) + ")";
      }

      auto size = static_cast<unsigned int>(16 * zoom);
      sf::Text label(*font, name, size);
      label.setFillColor(sf::Color(150, 150, 160));

      auto bounds = label.getLocalBounds();
      const sf::Vector2f& first = columns[c].slots[0].pos;
      label.setPosition({first.x - bounds.size.x / 2.0f,
                         first.y - nodeRadius - 34.0f});

      window.draw(label);
    }
  }

  window.display();
}
