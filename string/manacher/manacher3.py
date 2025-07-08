def manacher(s):
    l,r=-1,0
    z = [0]*len(s)
    for i in range(len(s)):
        j = 0
        if i < r:
            j = min(z[r-i+l], r-i)
        while i-j >= 0 and i+j < len(s) and s[i-j]==s[i+j]:
            j+=1
        if i+j > r:
            l,r = i-j, i+j
        z[i] = j
    return z