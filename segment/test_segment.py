from unittest import TestCase
import numpy as np
from segment5 import SegmentTree

class MyTest(TestCase):
    def test_segment(self):
        np.random.seed(42)
        a = np.random.randint(1,100,size=(1000,))
        t = SegmentTree(a)
        a[120] = 32 
        sum = 0
        for i in range(12,353):
            sum += a[i]
        t.update(120, 32)
        # self.assertEqual(t.sum(12,352), sum)
        self.assertEqual(t.query(12,352), sum)
