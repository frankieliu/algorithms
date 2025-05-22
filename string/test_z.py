from unittest import TestCase

from z import z

class TestZ(TestCase):
    def test_z(self):
        self.assertEqual(z("aaaaa"), [0,4,3,2,1])
        self.assertEqual(z("aaabaab"), [0,2,1,0,2,1,0])
        self.assertEqual(z("abacaba"), [0,0,1,0,3,0,1])