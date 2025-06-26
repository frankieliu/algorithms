from unittest import TestCase
from persistent import PersistentSegmentTree


class MyTestCase(TestCase):
    def test_feature(self):
        data = [1, 5, 2, 6, 3, 7, 4]
        mst = PersistentSegmentTree(data)
        self.assertEqual(mst.kth_smallest(1, 5, 3), 5)
