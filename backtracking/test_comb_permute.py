from unittest import TestCase
from permutation_repeated1 import permute_unique
from combination_repeated1 import unique_combinations
from subset import subsets
from subset_repeated1 import subsets_repeated

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

    def test_unique(self):
        input = [1,1,2,2,3,3,3,3]
        k = 3
        output = unique_combinations(input, k)
        expected = [[1, 1, 2], [1, 1, 3], [1, 2, 2], [1, 2, 3], [1, 3, 3], [2, 2, 3], [2, 3, 3], [3, 3, 3]]
        self.assertEqual(output, expected)

    def test_subsets(self):
        numbers = [1, 2, 3]
        output = subsets(numbers)
        expected = [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]
        self.assertEqual(output, expected)
    
    def test_subsets_repeated(self):
        nums = [1, 1, 2]
        output = subsets_repeated(nums)
        expected = [[], [2], [1], [1, 2], [1, 1], [1, 1, 2]]
        self.assertEqual(output, expected)