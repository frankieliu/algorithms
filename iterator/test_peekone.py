from unittest import TestCase
from peekone import Peakable

class TestPeekOne(TestCase):
    def test_peekable(self):
        p = Peakable(iter([1,2]))
        self.assertEqual(p.peek(),1)
        self.assertEqual(p.peek(),1)
        self.assertEqual(next(p),1)
        self.assertEqual(p.peek(),2)
        self.assertEqual(next(p),2)
        self.assertEqual(p.peek(),None)
        with self.assertRaises(StopIteration):
            next(p)