def z(s):
    l, r = -1, 0
    z = [0] * len(s)
    for i in range(1, len(s)):
        j = 0
        if i < r:
            j = min(z[i - l], r - i)
        while i + j < len(s) and s[i + j] == s[j]:
            j += 1
        if i + j > r:
            l, r = i, i + j
        z[i] = j
    return z
