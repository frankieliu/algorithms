from unittest import TestCase
import numpy as np
from bit3 import BIT
from bit_ru_pq import BIT_RangeUpdate_PointQuery
from bit_ru_rq import BIT_RangeUpdate_RangeQuery

class MyCase(TestCase):

    def test_feature(self):
        bit = BIT([0]*10)
        bit.update(3, 5)      # Add 5 at index 3
        bit.update(5, 2)      # Add 2 at index 5
        self.assertEqual(bit.query(5),7)   # Sum of indices 1 through 5 => 5 + 2 = 7
        self.assertEqual(bit.range_query(3, 5),7)  # Sum from index 3 to 5 => 5 + 2 = 7

    def test_feature2(self):
        a = np.random.randint(0,1000, size=(1000,))
        bit = BIT(list(a))
        s = 0
        for i in range(23, 754):
            s += a[i]
        self.assertEqual(bit.range_query(23, 753), s)
        
    def test_feature3(self):
        bit = BIT_RangeUpdate_PointQuery(10)
        bit.range_add(2, 5, 3)      # Add 3 to indices 2 through 5
        self.assertEqual(bit.point_query(4),3)   # Should print 3
        self.assertEqual(bit.point_query(6),0)   # Should print 0

    def test_feature4(self):
        bit = BIT_RangeUpdate_RangeQuery(10)
        bit.range_add(2, 5, 3)       # Add 3 to range [2, 5]
        self.assertEqual(bit.range_sum(1, 5), 12)   # Should print 3*4 = 12
        self.assertEqual(bit.range_sum(4, 6), 6)   # Should print 3*2 = 6
 