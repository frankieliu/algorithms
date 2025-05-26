def pi(s):
    n = len(s)
    pi = [0] * n
    for i in range(1, n):
        j = pi[i-1] # next char in the prefix
        while j > 0 and s[i] != s[j]:
            j = pi[j-1]
        if s[i] == s[j]:
            j += 1
        pi[i] = j
    return pi