# to remember
1. pi[i] is the length of the longest suffix ending at i
   that is a proper prefix
1. pi[i] if s[p[i-1]] == s[i]
1. if not, then the next possible candidate is
1. pi[pi[i-1]-1]

abc:::abc*......abc:::abc*
         ^
         pi[i-1]
        ^
        pi[i-1]-1
   ^
   pi[pi[i-1]-1]
