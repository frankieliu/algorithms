from unittest import TestCase
from kmp1 import pi

class TestKmp(TestCase):
    def test_kmp(self):
        self.assertEqual(pi("abcabcd"), [0,0,0,1,2,3,0])
        self.assertEqual(pi("aabaaab"), [0,1,0,1,2,2,3])