#ifndef LAYER_H
#define LAYER_H

#include "math/MatrixOps.h"

enum class Activation { ReLU, Softmax, Linear };

class Layer {
public:
  Layer(int in, int out, Activation activation);

  const Matrix&  getWeights() const {return this->W;}
  const Matrix&  getBias() const {return this->b;}
  const Matrix&  getInput() const {return this->ak;}
  const Matrix&  getPreAct() const {return this->z;}
  const Matrix&  getOutput() const {return this->aj;}
  Activation     getActivation() const {return this->activation;}

  void    setWeights(const Matrix& w);
  void    setBias(const Matrix& bias);

  void    forward(const Matrix& input);
  Matrix  backward(const Matrix& gradOutput);
  void    applyGradients(float learningRate, int count);
  void    zeroGradients();

private:
  Matrix W, dW, b, db, ak, z, aj;
  Activation activation;
};


#endif
