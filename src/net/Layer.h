#ifndef LAYER_H
#define LAYER_H

#include "math/MatrixOps.h"

class Layer {
public:
  Layer(int in, int out);

  Matrix  getWeights() const {return this->W;}
  Matrix  getBias() const {return this->b;}
  Matrix  getInput() const {return this->ak;}
  Matrix  getPreAct() const {return this->z;}
  Matrix  getOutput() const {return this->aj;}

  void    setWeights(const Matrix& w);
  void    setBias(const Matrix& bias);

  void    forward(const Matrix& input);
  Matrix  backward(const Matrix& gradOutput);
  void    applyGradients(float learningRate);

private:
  Matrix W, dW, b, db, ak, z, aj;
};


#endif
