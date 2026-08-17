#include "net/Layer.h"

Layer::Layer(int in, int out)
  : W(out, in), dW(out, in), b(out, 1), db(out, 1), ak(in, 1), z(out, 1), aj(out, 1)
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


void Layer::forward(const Matrix& input)
{
  this->ak = input;
  this->z = W * input + b;
  this->aj = ReLU(this->z);
}


Matrix Layer::backward(const Matrix& gradOutput)
{
  Matrix errOutput = hadamard(gradOutput, ReLUPrime(z));
  this->db = db + errOutput;
  this->dW = dW + errOutput * transpose(ak);
  return transpose(W) * errOutput;
}


void Layer::applyGradients(float learningRate)
{
  W = W - scalarMult(dW, learningRate);
  b = b - scalarMult(db, learningRate);
}


void Layer::zeroGradients()
{
  dW.zero();
  db.zero();
}
