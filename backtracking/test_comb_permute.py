from unittest import TestCase
from permutation_repetition1 import permute_unique


class MyTest(TestCase):
    def test_feature(self):
        expected = [[1, 2, 2, 3],
                    [1, 2, 3, 2],
                    [1, 3, 2, 2],
                    [2, 1, 2, 3],
                    [2, 1, 3, 2],
                    [2, 2, 1, 3],
                    [2, 2, 3, 1],
                    [2, 3, 1, 2],
                    [2, 3, 2, 1],
                    [3, 1, 2, 2],
                    [3, 2, 1, 2],
                    [3, 2, 2, 1]]
        nums = [1, 2, 2, 3]
        self.assertEqual(permute_unique(nums), expected)
