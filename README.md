<img width="1060" height="226" alt="image" src="https://github.com/user-attachments/assets/0a2f6cac-19ca-4c64-9512-089aac2bf07d" />

# Deep Learning from Scratch (DLFS)
Deep Learning from Scratch (DLFS) is an educational repository containing from scratch implementations of modern deep learning algorithms and architectures. It currently contains several core elements like the multi-layer perceptron (MLP), backpropagation and optimizers, with the goal of eventually building towards more complex architectures like convolutional neural networks and transformers. I am building DLFS to gain a more complete understanding of deep learning, however I hope the implementations and descriptions can be of some use to others.

For a complete step-by-step overview of the algorithms, see: TODO

## Core Elements
DLFS maintains 3 primary elements: a core library implementing fundamental deep learning data structures and algorithms, libraries for the key architectures encountered in deep learning, and examples which contains runnable training and evaluation implementations to demonstrate that the code works on real data.

The core data-structures implemented are the Tensor engine, the neural network (the building blocks used to compose MLPs), and trainers which are the glue holding the algorithms together. The trainer design is modular, allowing for custom optimizers, loss functions, MLP architectures, logging callbacks, and data handlers. The goal of this design is to encourage experimentation without having to rewrite major components from scratch.

The architecture libraries use the core building blocks (at times providing concrete interface implementations) to construct common deep-learning architectures. Currently, only linear regression is supported, however the baseline goal is to also support convolutional neural networks and the transformer. These architectures will provide a strong foundation for future experimentation and additional architectures down the line.

The examples directory contains runnable code which uses the core library and various architectures to solve real problems. It not only demonstrates how these tools can be practically used, but also acts as verification that they actually work.

## Run Examples
This repository uses [Nix](https://nixos.org/) for dependency management. When you enter a Nix development shell, the full set of dependencies will become available. The current set of supported examples are:
- Linear Regression
- TODO

To run an example, use the following commands:
```
# Enter Nix development shell
nix develop

# Run example
python -m examples.linear_regression
```

## Pytorch
While this repository is very helpful in building foundational knowledge, modern deep-learning research leverages popular frameworks such as [Pytorch](https://pytorch.org/) or [TensorFlow](https://www.tensorflow.org/) (among many other excellent options). My goal with this project is not to re-invent the wheel - it is to build the basics from scratch, gain a deeper understanding of how these packages operate under the hood, and compare my approaches with the top frameworks to find areas for optimization or improvement. Once this goal has been achieved, any further deep-learning research can and should leverage these rich resources.

The goal is to implement every example both from-scratch and using a Pytorch implementation on the same data, comparing accuracy of the results, speed of training and runtime inference and identifying ways to improve my solutions.
