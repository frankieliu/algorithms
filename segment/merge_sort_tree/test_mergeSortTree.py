from unittest import TestCase
from mergeSortTree import MergeSortTree


class MyTestCase(TestCase):
    def test_feature(self):
        data = [1, 5, 2, 6, 3, 7, 4]
        mst = MergeSortTree(data)
        self.assertEqual(mst.kth_smallest(1, 5, 3), 5)
