def z(s):
    l, r = 0, 0
    n = len(s)
    z = [0] * n
    for i in range(1, n):
        if i < r:
            z[i] = min(r-i, z[i-l])
        while i + z[i] < n and s[i+z[i]] == s[z[i]]:
            z[i]+=1
        if i + z[i] > r:
            l = i
            r = i + z[i]
    return z