import numpy as np

class Matrix:
    def __init__(self, data, _children=(), _op=''):
        """
        Initialize the Matrix with 2D data, optional children, and operation type.

        Args:
            data (array-like): Input data for the Matrix.
            _children (tuple): Optional tuple of child Matrixs for autograd.
            _op (str): Operation type that created this Matrix.
        """
        if not isinstance(_children, tuple):
            raise TypeError("_children must be a tuple")
        if not isinstance(_op, str):
            raise TypeError("_op must be a string")

        self.data = np.array(data, dtype=np.float32)
        if self.data.ndim == 0:
            self.data = self.data.reshape(1, 1)
        elif self.data.ndim == 1:
            self.data = self.data.reshape(-1, 1)
        assert self.data.ndim == 2, "Input must be a Matrix (1D or 2D Matrix)"


        self.grad = np.zeros_like(self.data, dtype=np.float32)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self._is_leaf = len(_children) == 0
        self._backward_called = False
        self.shape = self.data.shape

    def __repr__(self, print_grad = False):
        if print_grad:
            return f"Matrix(data=\n{self.data},\n grad=\n{self.grad}\n)"
        return f"Matrix(data=\n{self.data}\n)"


    def __call__(self):
        """
        Return the matrix data.

        Returns:
            numpy.ndarray: The data contained in the matrix.
        """
        return self.data

    def __eq__(self, other):
        """
        Compare this Matrix with another Matrix or array-like object.
        
        Args:
            other (Matrix or array-like): The object to compare with.
        
        Returns:
            bool: True if the data in both objects is equal, False otherwise.
        """
        if isinstance(other, Matrix):
            return np.array_equal(self.data, other.data)
        elif isinstance(other, (np.ndarray, list)):
            return np.array_equal(self.data, np.array(other))
        return False
    
    def __hash__(self):
        return id(self)

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
        """Reset the gradients of the Matrix to zero."""
        self.grad = np.zeros_like(self.data, dtype=np.float32)
        self._backward_called = False

    def __add__(self, other):
        """Element-wise addition of two matrices."""
        other = other if isinstance(other, Matrix) else Matrix(other)
        out = Matrix(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out
    
    def __sub__(self, other): 
        """Element-wise substraction of two matrices."""
        other = other if isinstance(other, Matrix) else Matrix(other)
        out = Matrix(self.data - other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad -= out.grad

        out._backward = _backward
        return out
    
    def __mul__(self, other):
        """Element-wise multiplication of two Matrix objects.
           or, Hadamard product 
        """
        other = other if isinstance(other, Matrix) else Matrix(other)
        out = Matrix(self.data * other.data, (self, other), '*')

        def _backward():
            grad_self = other.data * out.grad
            grad_other = self.data * out.grad

            self.grad += grad_self
            other.grad += grad_other

        out._backward = _backward
        return out

    def sum(self):
        """Sum all elements of the Matrix and return a scalar Matrix."""
        out = Matrix(np.sum(self.data).reshape(1, 1), (self,), 'sum')

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out
    
    def relu(self):
        """Apply the ReLU activation function element-wise."""
        out = Matrix(np.maximum(0, self.data), (self,), 'ReLU')
        out._op = 'ReLU'

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out
    
    def matmul(self, other):
        """Perform matrix multiplication."""
        other = other if isinstance(other, Matrix) else Matrix(other)
        assert self.data.shape[1] == other.data.shape[0], "Inner dimensions must match for matrix multiplication"

        out = Matrix(np.matmul(self.data, other.data), (self, other), 'matmul')

        def _backward():
            self.grad += np.matmul(out.grad, other.data.T)
            other.grad += np.matmul(self.data.T, out.grad)
        out._backward = _backward

        return out

    def __pow__(self, other):
        """Raise the Matrix elements to the power of other."""
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Matrix(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad

        out._backward = _backward
        return out

    def transpose(self):
        """Transpose the last two dimensions of the Matrix."""
        out = Matrix(self.data.T, (self,), 'transpose')

        def _backward():
            self.grad += out.grad.T
        out._backward = _backward

        return out

    @property
    def T(self):
        """Shortcut for transpose method."""
        return self.transpose()

    def __neg__(self): 
        """Element-wise negation of the Matrix."""
        return self * -1

    def __truediv__(self, other): 
        """Element-wise division of two Matrix objects with broadcasting support."""
        return self * other**-1

    def __radd__(self, other): 
        """Right-hand side addition for scalar + Matrix."""
        return self + other

    def __rsub__(self, other): 
        """Right-hand side subtraction for scalar - Matrix."""
        return other + (-self)

    def __rmul__(self, other): 
        """Right-hand side multiplication for scalar * Matrix."""
        return self * other

    def __rtruediv__(self, other): 
        """Right-hand side division for scalar / Matrix."""
        return other * self**-1
