# how to count the possible number of subarrays?

[1,2,3]

possible starting locations: n

possible ending locations: (n, n-1, ... 1) which on the average is (n+1)/2

1+2+3 = 6
3*4/2 = 6

So another 

LC 2962

Another way to count:
how many substrings can you make ending at index i, i+1 strings

Therefore for all endings:
1 + 2 + 3 + 4 + ... + n

= n (n+1) / 2