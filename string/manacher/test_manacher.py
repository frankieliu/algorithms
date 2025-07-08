from unittest import TestCase
from manacher3 import manacher

class TestManacher(TestCase):
    def testManacher(self):
        self.assertEqual(manacher("abcbcba"), [1,1,2,4,2,1,1])
        self.assertEqual(manacher("mississippi"), [1,1,1,1,4,1,1,1,1,1,1])
        self.assertEqual(manacher("ababacaca"), [1,2,3,2,1,2,3,2,1])
        self.assertEqual(manacher("aaaaa"), [1,2,3,2,1])
        self.assertEqual(manacher("#a#b#c#b#c#b#a#"),[1,2,1,2,1,4,1,8,1,4,1,2,1,2,1])