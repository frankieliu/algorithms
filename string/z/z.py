# z[i] longest match starting at i (prefix of suffix starting at i)
# matching prefix of s
# z: aaaaa => 01234
#
# keep the longest match
# [l,r)
# if i in l,r then a good guess is z(i-l) but until the edge r

def z(s):
    z = [0] * len(s)
    l, r = 0, 0   # make inclusive l and exclusive r
    for i in range(1, len(s)):
        # Use previous knowledge and limit to edge of r 
        if i < r:
            z[i] = min(r-i, z[i-l])
        # possibly extend z
        while i + z[i] < len(s) and s[i+z[i]] == s[z[i]]:
            z[i] += 1
        if i + z[i] > r:
            l, r = i, i + z[i]
    return z