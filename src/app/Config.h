#ifndef CONFIG_H
#define CONFIG_H

#include <string>

const bool TRAIN = false;

const int TRAIN_SIZE = 60000;
const int TEST_SIZE = 10000;
const int EPOCHS = 30;
const int BATCH_SIZE = 32;
const float LEARNING_RATE = 0.04f;

const std::string WEIGHTS_PATH = "weights.txt";

#endif
