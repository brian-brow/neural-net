#ifndef LAYER_H
#define LAYER_H

#include "math/MatrixOps.h"

class Layer {
public:
  Layer(int in, int out);

  Matrix getWeights() const {return this->W;}
  Matrix getBias() const {return this->b;}
  Matrix getInput() const {return this->input;}
  Matrix getPreAct() const {return this->preAct;}
  Matrix getOutput() const {return this->output;}

  void setWeights(const Matrix& w);
  void setBias(const Matrix& bias);

  void forward(const Matrix& input);
  void backward();

private:
  Matrix W, b, input, preAct, output;
};


#endif
