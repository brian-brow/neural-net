#ifndef NETWORK_H
#define NETWORK_H

#include <string>
#include <vector>
#include "math/MatrixOps.h"
#include "net/NetworkOps.h"
#include "net/Layer.h"

class Network {
public:
  Network(const std::vector<int>& widths);

  // Rebuild a trained network from a file written by save(). A named factory
  // rather than a constructor, because Network({784, 16, 10}) would otherwise
  // be ambiguous: std::string also accepts a braced list of numbers.
  // Throws std::runtime_error if the file is missing, malformed, or truncated.
  static Network load(const std::string& path);

  void save(const std::string& path) const;

  // False once any weight or bias has become nan or inf. Check before saving,
  // or a diverged run overwrites the last good weights with garbage.
  bool isFinite() const;

  void forward(const Matrix& input);
  void backward(const Matrix& target);

  void zero();
  void applyGradients(float learningRate, int count);

  const std::vector<Layer>& getLayers() const {return layers;}

private:
  Network() : LAYERS(0) {}   // only load() builds a Network without widths

  std::vector<Layer> layers;
  int LAYERS;
};


#endif
