# to remember
1. odd first, even add #a#b#c#c#b#a#
1. z[i] = radius of palindrome
1. (l,r) : make both sides exclusive easier
1. begin with 
1. if i in l,r, find the symmetric point
   j = min( z[l + r - i - 1], r-i )
   while s[i + j] == s[i - j] and boundary:
     j+=1

# hashing 
Hashing algo for palindrome:

https://baites.github.io/algorithms/algorithm-analysis/2020/03/23/string-hashing-and-palindromes.html
