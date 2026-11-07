import numpy as np


class Parameter:
    """Abstraction for defining model parameters"""

    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = np.zeros_like(data)

    def zero_grad(self) -> None:
        self.grad.fill(0)


class Neuron:
    """Representation of a neuron with weights, a bias and an activation function"""

    def __init__(self, dim: int):
        pass


class Layer:
    """Representation of a layer of a deep neural network"""

    def __init__(self, dim: int):
        pass


class MLP:
    """Representation of a multi-layer perceptron"""

    def __init__(self, dim: int):
        pass
