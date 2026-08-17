#include "net/Network.h"

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace {

// Bumped whenever the file layout changes, so an old file fails loudly
// instead of being read as garbage.
const char* MAGIC = "nnweights";
const int FORMAT_VERSION = 1;

}  // namespace


Network::Network(const std::vector<int>& widths)
{
  for (size_t i = 0; i < widths.size() - 1; i++) {
    if (i < widths.size() - 2) {
      this->layers.push_back(Layer(widths[i], widths[i+1], Activation::ReLU));
    } else {
      this->layers.push_back(Layer(widths[i], widths[i+1], Activation::Softmax));
    }
  }
  LAYERS = layers.size();
}


Network Network::load(const std::string& path)
{
  Network net;
  std::ifstream file(path);

  if (!file) {
    throw std::runtime_error("could not open weights file " + path);
  }

  std::string magic;
  int version = 0;
  file >> magic >> version;

  if (magic != MAGIC || version != FORMAT_VERSION) {
    throw std::runtime_error(path + " is not a version " +
                             std::to_string(FORMAT_VERSION) + " weights file");
  }

  int count = 0;
  file >> count;

  if (!file || count <= 0) {
    throw std::runtime_error("bad layer count in " + path);
  }

  for (int l = 0; l < count; l++) {
    int in = 0, out = 0, act = 0;
    file >> in >> out >> act;

    if (!file || in <= 0 || out <= 0) {
      throw std::runtime_error("bad header for layer " + std::to_string(l) +
                               " in " + path);
    }
    if (act < 0 || act > static_cast<int>(Activation::Linear)) {
      throw std::runtime_error("unknown activation " + std::to_string(act) +
                               " for layer " + std::to_string(l));
    }

    Matrix W(out, in);
    for (int i = 0; i < out; i++) {
      for (int j = 0; j < in; j++) {
        float v = 0.0f;
        file >> v;
        W.setAt(i, j, v);
      }
    }

    Matrix b(out, 1);
    for (int i = 0; i < out; i++) {
      float v = 0.0f;
      file >> v;
      b.setAt(i, 0, v);
    }

    // One check after the whole layer rather than per value, so a truncated
    // file is reported once instead of thousands of times.
    if (!file) {
      throw std::runtime_error("truncated data for layer " + std::to_string(l) +
                               " in " + path);
    }

    Layer layer(in, out, static_cast<Activation>(act));
    layer.setWeights(W);
    layer.setBias(b);

    net.layers.push_back(layer);
  }

  net.LAYERS = net.layers.size();

  return net;
}


bool Network::isFinite() const
{
  for (const Layer& l : layers) {
    const Matrix& W = l.getWeights();
    for (int i = 0; i < W.getRows(); i++) {
      for (int j = 0; j < W.getCols(); j++) {
        if (!std::isfinite(W.at(i, j))) {
          return false;
        }
      }
    }

    const Matrix& b = l.getBias();
    for (int i = 0; i < b.getRows(); i++) {
      if (!std::isfinite(b.at(i, 0))) {
        return false;
      }
    }
  }

  return true;
}


void Network::save(const std::string& path) const
{
  std::ofstream file(path);

  if (!file) {
    throw std::runtime_error("could not open " + path + " for writing");
  }

  // 9 significant digits round-trips a float exactly.
  file.precision(9);

  file << MAGIC << ' ' << FORMAT_VERSION << '\n';
  file << layers.size() << '\n';

  for (const Layer& l : layers) {
    const Matrix& W = l.getWeights();   // (out x in)
    const Matrix& b = l.getBias();

    file << W.getCols() << ' ' << W.getRows() << ' '
         << static_cast<int>(l.getActivation()) << '\n';

    for (int i = 0; i < W.getRows(); i++) {
      for (int j = 0; j < W.getCols(); j++) {
        file << W.at(i, j) << ' ';
      }
      file << '\n';
    }

    for (int i = 0; i < b.getRows(); i++) {
      file << b.at(i, 0) << ' ';
    }
    file << '\n';
  }

  if (!file) {
    throw std::runtime_error("failed while writing " + path);
  }
}


void Network::forward(const Matrix& input)
{
  layers[0].forward(input);
  for (int i = 1; i < LAYERS; i++) {
    layers[i].forward(layers[i - 1].getOutput());
  }
}


void Network::backward(const Matrix& target)
{
  Matrix prev = layers[LAYERS-1].backward(
                  crossEntropyPrime(layers[LAYERS-1].getOutput(), target));
  for (int i = LAYERS - 2; i >= 0; i--) {
    prev = layers[i].backward(prev);
  }
}


void Network::zero()
{
  for (int i = 0; i < LAYERS; i++) {
    layers[i].zeroGradients();
  }
}


void Network::applyGradients(float learningRate, int count)
{
  for (int i = 0; i < LAYERS; i++) {
    layers[i].applyGradients(learningRate, count);
  }
}
