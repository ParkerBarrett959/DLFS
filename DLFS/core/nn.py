import numpy as np


class Tensor:
    """Abstraction for defining model parameters"""

    def __init__(self, data: np.ndarray, _prev=()):
        """Input data is a matrix where rows correspond to elements of a batch and columns represent each feature in a single sample"""
        self.data = data
        self.grad = np.zeros_like(data)

        # autodiff graph variables
        self._backward = lambda: None
        self._prev = set(_prev)

    def zero_grad(self) -> None:
        self.grad.fill(0)

    def __add__(self, other: Tensor) -> Tensor:
        if self.data.shape != other.data.shape:
            raise ValueError(f"Expected {self.shape}, but got {other.shape}")

        # create output tensor
        out = Tensor(self.data + other.data, (self, other))

        # set backward pass function
        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Tensor) -> Tensor:
        if self.data.shape != other.data.shape:
            raise ValueError(f"Expected {self.shape}, but got {other.shape}")

        # create output tensor
        out = Tensor(self.data * other.data, (self, other))

        # set backward pass function
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __matmul__(self, other: Tensor) -> Tensor:
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"Expected {self.shape}, but got {other.shape}")

        # create output tensor
        out = Tensor(self.data @ other.data, (self, other))

        # set backward pass function
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def relu(self) -> Tensor:
        out = Tensor(np.maximum(0, self.data), (self,))

        # set backward pass function
        def _backward():
            self.grad += out.grad * (self.data > 0)

        out._backward = _backward
        return out


class Neuron:
    """Representation of a neuron with weights, a bias and an activation function"""

    def __init__(self, dim: int, activation: str = "relu"):
        # random initial weights, uniform distribution on range [0, 1]
        self.dim = dim
        rng = np.random.default_rng()
        self.weights = rng.random(dim)

        # zero initial bias
        self.bias = 0.0

        # activation function
        self.activation = activation if activation is not None else None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.shape != (self.dim, 1):
            raise ValueError(f"Expected shape {self.dim, 1}, but got {x.shape}")

        return ().relu()
        return


class Layer:
    """Representation of a layer of a deep neural network"""

    def __init__(self, dim: int):
        pass


class MLP:
    """Representation of a multi-layer perceptron"""

    def __init__(self, dim: int):
        pass
