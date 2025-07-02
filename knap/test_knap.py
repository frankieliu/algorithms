import re
from unittest import TestCase
from knap_zero import kp

# From at coder:
# https://atcoder.jp/contests/dp/tasks/dp_d

# 3 8 (3 elements 8 capacity)
weight_value_str = """\
3 30
4 50
5 60
"""

# 6 15 (3 elements 15 capacity)
wv2 = """\
6 5
5 6
6 4
6 6
3 5
7 2
"""

def read_input(s):
    res = []
    for line in re.split(r"\n", s):
        if line and line[0].isnumeric():
            res.append([int(x) for x in re.split(r"\s+", line)])
    return res 

class MyTestCase(TestCase):
    def test_myfeature(self):
        cap = 8 
        expect = 90
        wv = read_input(weight_value_str)
        res = kp(wv, cap)       
        self.assertEqual(res, expect)
    
    def test_myfeature2(self):
        cap = 15
        expect = 17 
        wv = read_input(wv2)
        res = kp(wv, cap)       
        expect = 17 
        self.assertEqual(res, expect)