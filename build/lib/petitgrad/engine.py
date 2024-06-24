import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        """
        Initialize the Tensor with data, optional children, and operation type.

        Args:
            data (array-like): Input data for the tensor.
            _children (tuple): Optional tuple of child tensors for autograd.
            _op (str): Operation type that created this tensor.
        """
        self.data = np.array(data, dtype=np.float32)
        if self.data.ndim == 0:
            self.data = self.data.reshape(1, 1, 1)
        elif self.data.ndim == 1:
            self.data = self.data.reshape(1, -1, 1)
        elif self.data.ndim == 2:
            self.data = self.data.reshape(1, *self.data.shape)

        self.grad = np.zeros_like(self.data, dtype=np.float32)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self._is_leaf = len(_children) == 0
        self._backward_called = False

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __call__(self):
        """
        Return the tensor's data.

        Returns:
            numpy.ndarray: The data contained in the tensor.
        """
        return self.data

    def backward(self, gradient=None):
        """
        Perform backpropagation to compute gradients.

        Args:
            gradient (array-like): Initial gradient to propagate. If None, uses ones.
        """
        if self._backward_called:
            raise RuntimeError("backward() has already been called on this graph.")
        
        if gradient is None:
            gradient = np.ones_like(self.data, dtype=np.float32)
        
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = gradient
        for v in reversed(topo):
            v._backward()

        self._backward_called = True

    def zero_grad(self):
        """Reset the gradients of the tensor to zero."""
        self.grad = np.zeros_like(self.data, dtype=np.float32)
        self._backward_called = False

    def __add__(self, other):
        """Element-wise addition of two tensors with broadcasting support."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        
        def _backward():
            grad_self = out.grad
            grad_other = out.grad

            # Reduce gradients by summing along the broadcasted dimensions
            if grad_self.shape != self.data.shape:
                grad_self = np.sum(grad_self, axis=tuple(i for i, dim in enumerate(self.data.shape) if dim == 1), keepdims=True)
            if grad_other.shape != other.data.shape:
                grad_other = np.sum(grad_other, axis=tuple(i for i, dim in enumerate(other.data.shape) if dim == 1), keepdims=True)

            self.grad += grad_self
            other.grad += grad_other

        out._backward = _backward
        return out

    def __mul__(self, other):
        """Element-wise multiplication of two tensors with broadcasting support."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            grad_self = other.data * out.grad
            grad_other = self.data * out.grad

            # Reduce gradients by summing along the broadcasted dimensions
            if grad_self.shape != self.data.shape:
                grad_self = np.sum(grad_self, axis=tuple(i for i, dim in enumerate(self.data.shape) if dim == 1), keepdims=True)
            if grad_other.shape != other.data.shape:
                grad_other = np.sum(grad_other, axis=tuple(i for i, dim in enumerate(other.data.shape) if dim == 1), keepdims=True)

            self.grad += grad_self
            other.grad += grad_other

        out._backward = _backward
        return out

    def sum(self):
        """Sum all elements of the tensor and return a scalar tensor."""
        out = Tensor(np.sum(self.data).reshape(1, 1, 1), (self,), 'sum')

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out
    
    def relu(self):
        """Apply the ReLU activation function element-wise."""
        out = Tensor(np.maximum(0, self.data), (self,), 'ReLU')
        out._op = 'ReLU'

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out
    
    def matmul(self, other):
        """Perform matrix multiplication of two rank-2 tensors."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.matmul(self.data[0], other.data[0]).reshape(1, self.data.shape[1], other.data.shape[2]), (self, other), 'matmul')

        def _backward():
            self.grad += np.matmul(out.grad[0], other.data[0].T).reshape(1, self.data.shape[1], self.data.shape[2])
            other.grad += np.matmul(self.data[0].T, out.grad[0]).reshape(1, other.data.shape[1], other.data.shape[2])
        out._backward = _backward

        return out

    def __pow__(self, other):
        """Raise the tensor elements to the power of other."""
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad

        out._backward = _backward
        return out

    def transpose(self):
        """Transpose the last two dimensions of the tensor."""
        out = Tensor(self.data.transpose(0, 2, 1), (self,), 'transpose')

        def _backward():
            self.grad += out.grad.transpose(0, 2, 1)
        out._backward = _backward

        return out

    @property
    def T(self):
        """Shortcut for transpose method."""
        return self.transpose()

    def __neg__(self): 
        """Element-wise negation of the tensor."""
        return self * -1

    def __sub__(self, other): 
        """Element-wise subtraction of two tensors with broadcasting support."""
        return self + (-other)

    def __truediv__(self, other): 
        """Element-wise division of two tensors with broadcasting support."""
        return self * other**-1

    def __radd__(self, other): 
        """Right-hand side addition for scalar + tensor."""
        return self + other

    def __rsub__(self, other): 
        """Right-hand side subtraction for scalar - tensor."""
        return other + (-self)

    def __rmul__(self, other): 
        """Right-hand side multiplication for scalar * tensor."""
        return self * other

    def __rtruediv__(self, other): 
        """Right-hand side division for scalar / tensor."""
        return other * self**-1
