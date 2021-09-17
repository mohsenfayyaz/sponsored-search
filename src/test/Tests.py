import unittest
from src.Trainer.SpanPooling import *


class TestSpanPooling(unittest.TestCase):
    def test_avg_pooling(self):
        avg_pooling_module = get_pooling_module("avg")
        span = torch.tensor([[[1, 1, 1], [2, 2, 2], [0, 0, 0]], [[1, 1, 1], [2, 2, 2], [3, 3, 3]]])
        attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        pooled_representation = avg_pooling_module(span, attention_mask)
        eq = torch.equal(pooled_representation, torch.tensor([[1.5, 1.5, 1.5], [2., 2., 2.]]))
        self.assertTrue(eq)

    def test_max_pooling(self):
        avg_pooling_module = get_pooling_module("max")
        span = torch.tensor([[[1, 1, 1], [2, 2, 2], [10, 10, 10]], [[1, 1, 1], [2, 2, 2], [3, 3, 3]]])
        attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        pooled_representation = avg_pooling_module(span, attention_mask)
        eq = torch.equal(pooled_representation, torch.tensor([[2., 2., 2.], [3., 3., 3.]]))
        self.assertTrue(eq)


if __name__ == '__main__':
    unittest.main()
