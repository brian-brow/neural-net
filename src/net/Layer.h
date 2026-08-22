#ifndef LAYER_H
#define LAYER_H

#include "math/TensorOps.h"

enum class Activation { ReLU, Softmax, Linear };

class Layer {
public:
  Layer(int in, int out, Activation activation);

  const Tensor&  getWeights() const {return this->W;}
  const Tensor&  getBias() const {return this->b;}
  const Tensor&  getInput() const {return this->ak;}
  const Tensor&  getPreAct() const {return this->z;}
  const Tensor&  getOutput() const {return this->aj;}
  Activation     getActivation() const {return this->activation;}

  void    setWeights(const Tensor& w);
  void    setBias(const Tensor& bias);

  void    forward(const Tensor& input);
  Tensor  backward(const Tensor& gradOutput);
  void    applyGradients(float learningRate, int count);
  void    zeroGradients();

private:
  Tensor W, dW, b, db, ak, z, aj;
  Activation activation;
};


#endif
