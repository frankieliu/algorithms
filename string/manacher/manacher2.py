def manacher(s):
    l, r = 0, 0
    z = [0] * len(s)
    for i in range(1, len(s)):
        j = 0
        if i <= r:
            j = min(z[r - i + l], r - i + 1)
        while i - j >= 0 and i + j < len(s) and s[i - j] == s[i + j]:
            j += 1
        z[i] = j - 1
        if i + z[i] > r:
            l = i - z[i]
            r = i + z[i]
    return [x + 1 for x in z]
