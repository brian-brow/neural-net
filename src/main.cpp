#include <iostream>
#include <stdexcept>
#include "app/Config.h"
#include "app/Train.h"
#include "app/Visualize.h"
#include "data/Mnist.h"
#include "math/Tensor.h"
#include "math/TensorOps.h"


int main()
{
  MnistSet test = loadMnist("data/t10k-images-idx3-ubyte",
                          "data/t10k-labels-idx1-ubyte", TEST_SIZE);

  if (TRAIN) {
    return trainAndSave(test);
  }

  try {
    return showOneDigit(test);
  } catch (const std::exception& e) {
    std::cout << e.what() << "\nRun once with TRAIN = true to create it.\n";
    return 1;
  }
}
