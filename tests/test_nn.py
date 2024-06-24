import unittest
import numpy as np
from petitgrad.nn import Layer, MLP

class TestNN(unittest.TestCase):
    def test_layer_creation(self):
        layer = Layer(10, 5)
        self.assertEqual(layer.weight.data.shape, (1, 10, 5))
        self.assertEqual(layer.bias.data.shape, (1, 1, 5))

    def test_layer_forward(self):
        layer = Layer(3, 2)
        x = np.random.randn(1, 4, 3)
        output = layer(x)
        self.assertEqual(output.data.shape, (1, 4, 2))

    def test_mlp_creation(self):
        mlp = MLP(10, [8, 6, 4])
        self.assertEqual(len(mlp.layers), 3)
        self.assertEqual(mlp.layers[0].weight.data.shape, (1, 10, 8))
        self.assertEqual(mlp.layers[1].weight.data.shape, (1, 8, 6))
        self.assertEqual(mlp.layers[2].weight.data.shape, (1, 6, 4))

    def test_mlp_forward(self):
        mlp = MLP(5, [4, 3])
        x = np.random.randn(1, 2, 5)
        output = mlp(x)
        self.assertEqual(output.data.shape, (1, 2, 3))

if __name__ == '__main__':
    unittest.main()
