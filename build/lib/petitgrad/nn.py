from petitgrad.engine import Tensor
import numpy as np

class Layer:
    def __init__(self, in_features, out_features):
        """
        Initialize a fully connected layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
        """
        self.weight = Tensor(np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features))
        self.bias = Tensor(np.zeros((1, out_features)))
    
    def __call__(self, x):
        """
        Perform the forward pass for the layer.

        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor after applying weights and bias.
        """
        if not isinstance(x, Tensor):
            x = Tensor(x)
        return x.matmul(self.weight) + self.bias
    
    def parameters(self):
        """
        Return the parameters of the layer.

        Returns:
            list: List containing the weight and bias tensors.
        """
        return [self.weight, self.bias]

class MLP:
    def __init__(self, nin, nouts):
        """
        Initialize a multi-layer perceptron.

        Args:
            nin (int): Number of input features.
            nouts (list of int): List of output sizes for each layer.
        """
        self.layers = []
        sz = [nin] + nouts
        for i in range(len(nouts)):
            layer = Layer(sz[i], sz[i+1])
            self.layers.append(layer)
    
    def __call__(self, x):
        """
        Perform the forward pass for the MLP.

        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor after passing through all layers.
        """
        if not isinstance(x, Tensor):
            x = Tensor(x)
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = out.relu()
        return out
    
    def parameters(self):
        """
        Return the parameters of all layers in the MLP.

        Returns:
            list: List of tensors containing all weights and biases.
        """
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
