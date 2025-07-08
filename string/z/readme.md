# To remember:
1. z[i] is the length of the longest substring starting at i
   which is also a prefix of s
1. [lo,hi) is the rightmost
1. j = 0 : use this to match manacher
   if i is within [lo,hi):
      j = min(r-i, z[i-l])
      z[i] is related to z[i-lo]
   while s[i+j] == s[j], j+=1
   if i+j > r:
      l,r = ..