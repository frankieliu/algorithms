from unittest import TestCase
from bit import BIT

class MyCase(TestCase):
    def test_feature(self):
        bit = BIT(10)
        bit.update(3, 5)      # Add 5 at index 3
        bit.update(5, 2)      # Add 2 at index 5
        self.assertEqual(bit.query(5),7)   # Sum of indices 1 through 5 => 5 + 2 = 7
        self.assertEqual(bit.range_query(3, 5),7)  # Sum from index 3 to 5 => 5 + 2 = 7
