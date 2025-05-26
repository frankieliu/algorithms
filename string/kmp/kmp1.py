def pi(s):
    n = len(s)
    pi = [0]*n
    for i in range(1, n):
        j = pi[i-1]
        while j >0 and s[j] != s[i]:
            j = pi[j-1]
        if s[j] == s[i]:
            j += 1
        pi[i] = j
    return pi