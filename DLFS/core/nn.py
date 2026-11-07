import numpy as np
from .engine import Tensor

class Layer:
    """Representation of a neural net layer. Each column of W represents weights of a single neuron and each column of B represents the bias of a single neuron. Currently the layer always applies a relu activation function."""

    def __init__(self, nin: int, nout: int):
        # random initial weights and zero bias
        self.W = Tensor(np.random.randn(nin, nout) * np.sqrt(2.0 / nin))
        self.b = Tensor(np.zeros((1, nout)))

    def forward(self, X: Tensor) -> Tensor:
        return (X @ self.W + self.b).relu()

    def parameters(self):
        return [self.W, self.b]


class MLP:
    """Representation of a multi-layer perceptron"""

    def __init__(self, nin: int, nouts: list[int]):
        """nouts is the output of each layer from input to output"""
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def forward(self, X) -> Tensor:
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
