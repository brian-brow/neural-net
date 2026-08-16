#include "net/Layer.h"

Layer::Layer(int in, int out)
  : W(out, in), b(out, 1), input(in, 1), preAct(out, 1), output(out, 1)
{
  W.randomize();
}


void Layer::setWeights(const Matrix& w)
{
  assert(w.getRows() == W.getRows() && w.getCols() == W.getCols());
  this->W = w;
}


void Layer::setBias(const Matrix& _b)
{
  assert(_b.getRows() == b.getRows() && _b.getCols() == b.getCols());
  this->b = _b;
}


void Layer::forward(const Matrix& _input)
{
  this->input = _input;
  this->preAct = W * input + b;
  this->output = ReLU(this->preAct);
}

