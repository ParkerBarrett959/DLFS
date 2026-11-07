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

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def zero_grad(self) -> None:
        self.grad.fill(0)

    def __add__(self, other: "Tensor") -> "Tensor":
        # create output tensor
        out = Tensor(self.data + other.data, (self, other))

        # set backward pass function
        def _backward():
            if self.shape == out.shape:
                self.grad += out.grad
            else:
                axes = tuple(
                    i
                    for i, (d1, d2) in enumerate(zip(self.shape, out.shape))
                    if d1 != d2
                )
                self.grad += out.grad.sum(axis=axes, keepdims=True)

            if other.shape == out.shape:
                other.grad += out.grad
            else:
                diff = len(out.shape) - len(other.shape)
                axes = list(range(diff)) + [
                    i + diff
                    for i, (d1, d2) in enumerate(zip(other.shape, out.shape[diff:]))
                    if d1 != d2
                ]
                other.grad += out.grad.sum(axis=tuple(axes), keepdims=True)

        out._backward = _backward
        return out

    def __mul__(self, other: "Tensor") -> "Tensor":
        # create output tensor
        out = Tensor(self.data * other.data, (self, other))

        # set backward pass function
        def _backward():
            g1 = out.grad * other.data
            g2 = out.grad * self.data

            if self.shape == out.shape:
                self.grad += g1
            else:
                axes = tuple(
                    i
                    for i, (d1, d2) in enumerate(zip(self.shape, out.shape))
                    if d1 != d2
                )
                self.grad += g1.sum(axis=axes, keepdims=True)

            if other.shape == out.shape:
                other.grad += g2
            else:
                diff = len(out.shape) - len(other.shape)
                axes = list(range(diff)) + [
                    i + diff
                    for i, (d1, d2) in enumerate(zip(other.shape, out.shape[diff:]))
                    if d1 != d2
                ]
                other.grad += g2.sum(axis=tuple(axes), keepdims=True)

        out._backward = _backward
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        if self.shape[1] != other.shape[0]:
            raise ValueError(
                f"Expected inner dimensions to match, but got {self.shape} and {other.shape}"
            )

        # create output tensor
        out = Tensor(self.data @ other.data, (self, other))

        # set backward pass function
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
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
            node._backward()
