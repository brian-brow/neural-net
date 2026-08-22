#include "net/Layer.h"


Layer::Layer(int in, int out, Activation activation)
  : W({out, in}), dW({out, in}), b({out, 1}), db({out, 1}),
    ak({in, 1}), z({out, 1}), aj({out, 1})
{
  this->activation = activation;
  W.randomize();
}


void Layer::setWeights(const Tensor& w)
{
  assert(w.getDims() == W.getDims());
  this->W = w;
}


void Layer::setBias(const Tensor& _b)
{
  assert(_b.getDims() == b.getDims());
  this->b = _b;
}


void Layer::forward(const Tensor& input)
{
  this->ak = input;
  this->z = TensorOps::matmul(W, input) + b;
  if (activation == Activation::ReLU) {
    this->aj = TensorOps::ReLU(this->z);
  } else {
    this->aj = TensorOps::softmax(this->z);
  }
}


Tensor Layer::backward(const Tensor& gradOutput)
{
  Tensor errOutput(gradOutput.getDims());
  if (activation == Activation::ReLU) {
    errOutput = gradOutput * TensorOps::ReLUPrime(z);
  } else {
    errOutput = gradOutput;
  }
  this->db = db + errOutput;
  this->dW = dW + TensorOps::matmul(errOutput, ak.transpose());
  return TensorOps::matmul(W.transpose(), errOutput);
}


void Layer::applyGradients(float learningRate, int count)
{
  assert(count > 0);

  float scale = learningRate / count;

  W = W - dW * scale;
  b = b - db * scale;
}


void Layer::zeroGradients()
{
  dW.zero();
  db.zero();
}
