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

    def backward(self) -> None:
        """Build topographic list of all nodes in computation graph"""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # set gradient of output node wrt loss to 1
        self.grad = np.ones_like(self.data)

        # traverse through topographic list applying backward ops
        for node in reversed(topo):
            nod._backward()


class Layer:
    """Representation of a neural net layer. Each column of W represents weights of a single neuron and each column of B represents the bias of a single neuron. Currently the layer always applies a relu activation function."""

    def __init__(self, nin: int, nout: int):
        # random initial weights and zero bias
        self.W = Tensor(np.random.randn(nin, nout) * 0.1)
        self.b = Tensor(np.zeros((1, nout)))

    def forward(self, X: Tensor) -> Tensor:
        if X.data.shape[1] != self.W.shape[0]:
            raise ValueError(
                f"Expected X to have {self.W.shape[0]} columns, but had {X.data.shape[1]}"
            )

        # process layer
        return Tensor(X @ self.W + self.b).relu()

    def parameters(self):
        return [self.W, self.b]


class MLP:
    """Representation of a multi-layer perceptron"""

    def __init__(self, nin: int, nouts: list[int]):
        """nouts is the output of each layer from input to output"""
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def forward(self, X) -> Tensor:
        if X.data.shape[1] != self.layers[0].W.shape[0]:
            raise ValueError(
                f"Expected X to have {self.layers[0].W.shape[0]} columns, but had {X.data.shape[1]}"
            )

        for layer in self.layers:
            X = layer.forward(X)

        return X

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
