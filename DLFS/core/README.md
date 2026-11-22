# DLFS Core
The core library implements re-usable building block elements for deep-learning. Examples include data loaders, the tensor data structure, the multi-layer perceptron, and more. It also provides an interface for key components of deep-learning systems, like loss functions and optimizers. The goal of this README is to describe the architecture of DLFS core and the design of the key components.

## Tensor
The tensor data structure (defined in [engine.py](https://github.com/ParkerBarrett959/DLFS/blob/main/DLFS/core/engine.py)) is the fundamental building block of the deep-learning framework. Each tensor holds data, a gradient, a function object, and the set of tensors which were used to create it. The data and gradient are numpy arrays which leverages the efficiency of the matrix operations described in future steps. Each row corresponds to a unique input or data element, while columns correspond to the dimensionality of each sample. The function object stores a backward pass function, which is null by default.

The tensor data structure is important because it automatically builds a computation graph during the forward pass steps of the neural network training, which gets used in the backward pass to compute gradients. This algorithm is commonly referred to as automatic-differentiation. The tensor provides a closed-set of mathematical operations, for example, addition, subtraction, multiplication, exponentiation, etc., which are applied between two tensors and return another tensor. It also has activation functions, for example, the rectified-linear unit (relu). These simple operations can be chained together to build more complex expressions, leading to neural networks capable of representing complex, nonlinear functions. An example of a simple computation graph is shown below.

Insert computation graph...

Each tensor also has a backward() function which can be used to recursively build a topographical list of all nodes in the computation graph. This list is important for model training, because it allows us to walk backwards through the computation graph, from the end to the start, computing gradients along the way. This algorithm is called back-propagation.

Insert more details..maybe redo this section

## Neuron
The neuron is the atomic unit of the neural network. Each neuron is a linear combination of the inputs with an added bias. To achieve nonlinear approximations, an "activation function" is typically used on the output of the linear combination. The following figure shows the architecture of a single neuron.

Insert neuron image...

Each neuron has a set of weights, equal to the dimension of the input. For example, if an input vector has 3 features, $x_1$, $x_2$, and $x_3$, then the neuron will have 3 weights, $w_1$, $w_2$, and $w_3$. Each weight is multiplied by the corresponding input element, summed and a bias, $b$, is added. The linear portion of the neuron can be represented in matrix notation as:

$$
y = \mathbf{x} \mathbf{w} + b = \begin{bmatrix}x_1 \\ x_2 \\ x_3 \end{bmatrix} \begin{bmatrix}w_1 \\\\ w_2 \\\\ w_3 \end{bmatrix} + b
$$

The activation function, $\mathbf{f}$, is applied to the output. Many activation functions have been studied, however the rectified linear unit (relu) is one of the most common. The relu function is shown below:

Insert relu function image...

The output of the neuron, $\mathbf{o}$, is then given by:

$$
\mathbf{o} = \mathbf{f}(y) = \mathbf{f}(\mathbf{x} \mathbf{w} + b)
$$

## Layer


## Multi-Layer Perceptron
The multi-layer perceptron is 

## Data Loader
Insert...

## Trainer
Insert...

### Forward Pass
Insert...

### Loss
Insert...

### Backwards Pass
Insert...

### Optimizer
Insert...
