# PyTorch C++ API Examples

This directory contains C++ example applications that demonstrate the PyTorch C++ API (LibTorch).

## Prerequisites

- [LibTorch](https://pytorch.org/get-started/locally/) downloaded and extracted
- CMake 3.18 or newer
- A C++17-compatible compiler (GCC 9+ on Linux, MSVC 2019+ on Windows); the minimal example requires C++20

## Examples

### Minimal Example (`minimal/`)

A minimal application that creates a random tensor and prints it. This corresponds
to the example described in the [Installing C++ Distributions of PyTorch](https://pytorch.org/cppdocs/installing.html) guide.

**Build:**
```bash
cd minimal
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
cmake --build . --config Release
./example-app
```

### MNIST Training (`mnist/`)

A complete end-to-end example that defines and trains a simple fully-connected
neural network on the [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten
digits dataset. This corresponds to the example described in the
[C++ Frontend](https://pytorch.org/cppdocs/frontend.html) guide.

**Prepare MNIST data:**

Download the MNIST binary files from http://yann.lecun.com/exdb/mnist/ and
place them in a `data/` subdirectory inside the build folder:

```
build/
  data/
    train-images-idx3-ubyte
    train-labels-idx1-ubyte
    t10k-images-idx3-ubyte
    t10k-labels-idx1-ubyte
```

**Build:**
```bash
cd mnist
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
cmake --build . --config Release
mkdir data
# copy MNIST binary files into data/
./mnist
```
