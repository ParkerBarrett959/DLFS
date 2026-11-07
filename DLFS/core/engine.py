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

    @staticmethod
    def _unbroadcast(grad: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Sums out broadcasted dimensions so grad matches target_shape."""
        if grad.shape == target_shape:
            return grad

        # Handle 0-D scalar gradients
        if grad.ndim == 0:
            return np.full(target_shape, grad.item(), dtype=np.float32)

        # 1. Sum out extra leading dimensions if grad has more dims than target
        ndims_added = grad.ndim - len(target_shape)
        for _ in range(ndims_added):
            grad = grad.sum(axis=0)

        # 2. Sum out dimensions where target_shape has size 1 but grad has size > 1
        for i, dim in enumerate(target_shape):
            if dim == 1 and grad.shape[i] > 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad.reshape(target_shape)

    def __add__(self, other: "Tensor") -> "Tensor":
        # create output tensor
        out = Tensor(self.data + other.data, (self, other))

        # set backward pass function
        def _backward():
            self.grad += self._unbroadcast(out.grad, self.shape)
            other.grad += self._unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def sum(self) -> "Tensor":
        out = Tensor(np.array(np.sum(self.data)), (self,))

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other: "Tensor") -> "Tensor":
        # create output tensor
        out = Tensor(self.data - other.data, (self, other))

        # set backward pass function
        def _backward():
            self.grad += self._unbroadcast(out.grad, self.shape)
            other.grad += self._unbroadcast(-out.grad, other.shape)

        out._backward = _backward
        return out

    def __mul__(self, other: "Tensor") -> "Tensor":
        # handle tensor<->scalar multiplication
        if isinstance(other, (int, float, np.number)):
            out = Tensor(self.data * other, (self,))

            def _backward():
                self.grad += self._unbroadcast(out.grad * other, self.shape)

            out._backward = _backward
            return out

        # create output tensor
        out = Tensor(self.data * other.data, (self, other))

        # set backward pass function
        def _backward():
            g1 = out.grad * other.data
            g2 = out.grad * self.data
            self.grad += self._unbroadcast(g1, self.shape)
            other.grad += self._unbroadcast(g2, other.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other) -> "Tensor":
        return self * other

    def __truediv__(self, other) -> "Tensor":
        if isinstance(other, (int, float, np.number)):
            out = Tensor(self.data / other, (self,))

            def _backward():
                self.grad += self._unbroadcast(out.grad / other, self.shape)

            out._backward = _backward
            return out

        out = Tensor(self.data / other.data, (self, other))

        def _backward():
            g1 = out.grad / other.data
            g2 = -out.grad * self.data / (other.data**2)
            self.grad += self._unbroadcast(g1, self.shape)
            other.grad += self._unbroadcast(g2, other.shape)

        out._backward = _backward
        return out

    def __rtruediv__(self, other) -> "Tensor":
        if isinstance(other, (int, float, np.number)):
            out = Tensor(other / self.data, (self,))

            def _backward():
                self.grad += self._unbroadcast(
                    -out.grad * other / (self.data**2), self.shape
                )

            out._backward = _backward
            return out
        return Tensor(other) / self

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

    def __pow__(self, other) -> "Tensor":
        # create output tensor (other should be an int or float)
        out = Tensor(self.data**other, (self,))

        # set backward pass function
        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

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
