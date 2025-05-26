def manacher(s):
    n = len(s) 
    p = [0] * n 
    l,r=-1,0
    for i in range(n):
        if i < r:
            p[i] = min(r-i, p[l+r-i])
        while i + p[i] < len(s) and i - p[i] >= 0 and s[i+p[i]] == s[i-p[i]]:
            p[i] += 1
        if i + p[i] > r:
            l = i - p[i]
            r = i + p[i]
    return p
