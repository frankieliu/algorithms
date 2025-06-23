
"""
010110 
101010 (negative of the above)
    ^  will hit the lowest 1 set

r & -r
0000000000000001
00000001
0001    0001    0001
01  01  01  01  01
101010101010101010101

2 covers 1-2
4 covers 1-4
8 covers 1-8

Say I want from sum from 1-10
10 is written as 2+8

1010 Get bit at d10
  ^  Subtract r & -r
^    Get bit at d8

Going the other way:

Want to add at 10 -- this will affect things bigger than it
 1010 - 10

 1100 - 12
10000 - 16

how to start with an array:
 first element
 
"""
class Solution:
    # bit
    # 1011 & 0101
    #    0
    # 10111 & 01001
    # 10000 
    # 0 1 2 3 4 5 6 7 8
    #   1   1   1   1
    #   2 2     2 2 
    #   4 4 4 4
    #   8 8 8 8 8 8 8 8
    # 1000
    # should cover from g(8)+1 to 8  g(8) = 0
    # 0111
    # should cover from g(7)+1 to 7  g(7) = 110
    # to get the sum need to sum(g(7)) = sum(6), etc
    
    def add(self, b, r, val):
        lb = len(b)
        while r <= len(b):
            b[r] = max(b[r],val)
            r += (r & -r)

    def fbit(self, b, n):
        res = -1 
        while r > 0:
            res = max(res, b(r))
            r -= (r & -r)

    def fullbit(b, array):
        for i in range(1, len(array)+1):
            b[i] = max(b[i], array[i-1])
            ni = i + (i & -i)
            if ni <= len(array):
                b[ni] = max(b[ni], b[i])

    def findBuildings(self, heights: List[int]) -> List[int]:
        h = heights[::-1]
        msf = h[0]
        res = [0]
        for i in range(1, len(h)):
            if h[i] > msf:
                res.append(i)
                msf = h[i]
        return [len(h)-1-x for x in res][::-1]