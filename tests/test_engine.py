import unittest
import numpy as np
from petitgrad.engine import Tensor

class TestTensor(unittest.TestCase):
    def test_tensor_creation(self):
        t = Tensor([1, 2, 3])
        self.assertIsInstance(t, Tensor)
        self.assertEqual(t.data.shape, (1, 3, 1))

    def test_arithmetic_operations(self):
        a = Tensor(np.random.randn(2, 3, 4))
        b = Tensor(np.random.randn(1, 3, 1))
        self.assertEqual((a + b).data.shape, (2, 3, 4))
        self.assertEqual((a - b).data.shape, (2, 3, 4))
        self.assertEqual((a * b).data.shape, (2, 3, 4))
        self.assertEqual((a / b).data.shape, (2, 3, 4))

    def test_empty_tensor(self):
        a = Tensor(np.random.randn(2, 3, 4))
        c = Tensor(np.array([]))
        with self.assertRaises(Exception):
            a + c

    def test_large_tensor_operation(self):
        d = Tensor(np.random.randn(100, 100, 1))
        e = Tensor(np.random.randn(1, 100, 100))
        result = d + e
        self.assertEqual(result.data.shape, (100, 100, 100))

    def test_invalid_broadcasting(self):
        a = Tensor(np.random.randn(2, 3, 4))
        f = Tensor(np.random.randn(2, 3, 4, 5))
        with self.assertRaises(Exception):
            a + f

    def test_scalar_operations(self):
        a = Tensor([2.0])
        b = a * 2 + 1
        self.assertEqual(b.data.shape, (1, 1, 1))
        np.testing.assert_almost_equal(b.data, np.array([[[5.0]]]))

    def test_backward_pass(self):
        x = Tensor([2.0])
        y = x * x + x
        y.backward()
        self.assertEqual(x.grad.shape, (1, 1, 1))
        np.testing.assert_almost_equal(x.grad, np.array([[[5.0]]]))

if __name__ == '__main__':
    unittest.main()
