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

The scalar output of the neuron, $\mathbf{o}$, is then given by:

$$
\mathbf{o} = \mathbf{f}(y) = \mathbf{f}(\mathbf{x} \mathbf{w} + b)
$$

## Layer
If a neuron is the atomic unit of the neural network, then a layer is just a stack of neurons. It takes inputs from the previous layer and passes its outputs to the next.

Insert layer image...

The figure above shows what is referred to as a "fully-connected layer" - this means every input coming from the previous layer is fed into every neuron in the current layer. Rather than looping over each neuron in the layer and calculating its activation, it is more convenient instead to leverage the speed of matrix multiplication. For this reason, the concept of an individual neuron, while helpful to understand, is not actually implemented in the core library. Only the layer, which parallelizes the individual neuron computations, is implemented. Each input (scalar output of the previous layer) is stacked horizontally, while the weights and biases from each neuron of the current layer are also stacked horizontally.

$$
\mathbf{y} = \begin{bmatrix}y_1 \\ y_2 \\ \cdots \\ y_{n_{out}} \end{bmatrix} = \begin{bmatrix}x_1 \\ x_2 \\ \cdots \\ x_{n_{in}} \end{bmatrix} \begin{bmatrix} \mathbf{w}_1 \\ \mathbf{w}_2 \\ \cdots \\ \mathbf{w}_{n_{out}} \end{bmatrix} + \begin{bmatrix} \mathbf{b}_1 \\ \mathbf{b}_2 \\ \cdots \\ \mathbf{b}_{n_{out}} \end{bmatrix}
$$

Since every input, $x$, is a scalar, and every weight vector, $\mathbf{w}$ is a column vector, this effectively multiplies each input by the corresponding weight in each of the neurons in the current layer. The activation function can then be applied element-wise to the output, which returns the output of the given layer.

$$
\mathbf{o} = \mathbf{f}(y)
$$

Since this output is once again a set of scalars stacked horizontally, it can then be passed directly to the next layer.

One final performance optimization is the batching of layer operations. In our training loop, the goal is to pass a batch of data (with B samples) through each layer sequentially, rather than one at a time. This can be achieved by stacking each training sample (or output of the previous layer) vertically into one large matrix operation.

$$
\mathbf{Y} = \begin{bmatrix}\mathbf{y}^1 \\\\ \mathbf{y}^2 \\\\ \vdots \\\\ \mathbf{y}^{B} \end{bmatrix} = \begin{bmatrix}\mathbf{x}^1 \\\\ \mathbf{x}^2 \\\\ \vdots \\\\ \mathbf{x}^{B} \end{bmatrix} \begin{bmatrix} \mathbf{w}_1 \\ \mathbf{w}_2 \\ \cdots \\ \mathbf{w}_{n_{out}} \end{bmatrix} + \begin{bmatrix} \mathbf{b}_1 \\ \mathbf{b}_2 \\ \cdots \\ \mathbf{b}_{n_{out}} \end{bmatrix}
$$

Note that while the inputs are stacked, the weights and bias remain the same (horizontally stacked). The bias row vector is broadcast vertically, applying the same bias to every row, corresponding to every sample. The activation function is then applied once again elementwise to the output.

$$
\mathbf{O} = \mathbf{f}(Y)
$$

## Multi-Layer Perceptron
A multi-layer perceptron is just a series of layers, one after another. The first is called the input layer and the last is called the output layer. Each internal layer is called hidden. Each layer can have different numbers of neurons from the layers before and after it. An example of a multi-layer perceptron is shown below.

Insert MLP figure...

With the bulk of the complexity baked into the vectorized operations of the layer, the multi-layer perceptron implementation is very simple. The user must only specify the input dimension and the output dimension of each layer. At training and inference time, each layer is iterated over from start to finish, efficiently feeding the batched input data from the beginning of the network to the end.

## Data Loader
The data loader is a simple class for shuffling and providing batches of data to the main training algorithm. It holds a dataset, which is an abstract class implemented by specific algorithms or classes of problems. It also provides a dunder method which makes the dataset iterable. 

When constructed in a trainer object (described below), the user specifies the batch size for the data loader and whether the data should be shuffled. During each epoch of training, the data loader randomly shuffles the dataset and creates batches of the requested size. The trainer iterates over each of those batches in succession, eventually completing a single epoch. The shuffling and iteration process is repeated for every epoch in the model training process.

## Trainer
The trainer is the primary class tying the core library together. It holds a model (the neural network architecture), a loss criterion, an optimizer, and a set of injectable callbacks. The advantage of the trainer design is that the model, loss and optimizers are all abstract classes with a defined interface. This makes it simple to swap in and mix concrete implementations for each of these components, without having to rewrite everything from scratch every time.

The trainer class provides a fit function, which as the name suggest, "fits" a model to the training data - this process is known as model training. The trainer iterates over the number of epochs provided by the user, a hyperparameter of the training process. In each epoch, the training data is shuffled and split into batches by the Data Loader. For each batch of data, the gradients stored in the optimizer are reset to 0, the batch is passed forward through the model, the loss criterion is calculated, the backwards pass occurs which computes the gradients of each weight and bias in the model with respect to the loss, and finally the optimizer steps the model parameters in the direction of the gradient. The following diagram should the flow of a single epoch of training.

Insert diagaram...

Details of each step of the training epoch is provided below.

### Forward Pass
Insert...

### Loss
Insert...

### Backwards Pass
Insert...

### Optimizer
Insert...
