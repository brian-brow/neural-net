# neural-net

A basic Neural Network Implemented in C++

Years ago I tried to make a Neural Network in js and I failed miserably.
Out of rage now I avenge myself, conquering the vile beast known as backpropagation.
Some elements of this like the visualizer, data loaders, and saving weights to files features are nearly 100% vibe-coded.
That is because I do not care about that, I focused entirely on implementing the math and architecture.

Ideally in the future this will be developed into pytorch if it were much worse, and even run on GPU.
For now enjoy the beauty of training weights and looking at pictures of numbers labled correctly.

97.96% on the MNIST test set. No math libraries, no ML frameworks.

## Requirements

* g++ with C++20 (GCC 10+)
* **SFML 3.x**, not 2.x. The APIs are different and 2.x will not compile.
* GNU make, curl, gunzip

Arch: `pacman -S sfml gcc make`. Any Linux should work. macOS needs Homebrew
paths in the Makefile. Windows needs WSL.

## Setup

```sh
./scripts/fetch-mnist.sh    # downloads MNIST into data/, verifies checksums
make                        # builds ./nn
```

Other targets: `make run`, `make release`, `make clean`.

## Running

Mode is a constant at the top of `src/main.cpp`. It is compile-time, so
rebuild after changing it.

```cpp
const bool TRAIN = true;
```

**true** trains from scratch and writes `weights.txt` every epoch. No window.
About 20 minutes.

```
epoch 0    cost 0.19423    acc 94.28%
...
epoch 29   cost 0.0864172  acc 97.96%
```

**false** loads `weights.txt` and opens the visualizer on a random test digit.
Re-run for a different one. Wrong predictions are flagged.

```
loaded weights.txt   cost 0.0864172   acc 97.96%
test[7971]   actual 8   predicted 8   confidence 100%
```

## Model

```
784 -> 128 -> 64 -> 10
       ReLU   ReLU  Softmax
```

| | |
|---|---|
| Train / test | 60,000 / 10,000 |
| Epochs | 30 |
| Batch size | 32 |
| Learning rate | 0.04 |
| Loss | Cross-entropy |
| Init | He normal, bias zero |
| Optimiser | Mini-batch SGD, reshuffled each epoch |

Constants are at the top of `src/main.cpp`.

## Visualizer

SFML. `Esc` or `Q` to quit, resizable.

Left click anywhere for a random test image. The box at the bottom jumps to a
specific one: type a number, `Enter`. Out-of-range entries flash red and are
ignored. The box always shows the image currently on screen.

Input drawn as a 28x28 grid on the left. Nodes shaded by activation, black to
white. Weights are lines, red negative through grey to blue positive, thickness
by magnitude. Layers over 16 nodes show the first and last 8 with an ellipsis
between. The target sits detached on the right in gold.

## Layout

```
src/
  main.cpp        training loop, evaluation, mode switch
  math/           Matrix, MatrixOps
  net/            Layer, Network, NetworkOps
  data/           Mnist loader
  viz/            Visualizer
scripts/
  fetch-mnist.sh
```

Dependencies point one way: `math <- net <- viz`, with `main` composing them.

## Weights file

Plain text, 9 significant digits so floats round-trip exactly. Fully describes
the architecture, so `Network::load` does not need the widths.

```
nnweights 1        magic + version
3                  layer count
784 128 0          in, out, activation (0=ReLU 1=Softmax 2=Linear)
<W row-major>
<b>
```

## TODO

* Validation split
* Save the best network, not the most recent
* L2 or dropout, test cost bottoms out around epoch 9 and rises after
* Column-batching, `(784 x batch)` instead of one at a time
* CUDA
