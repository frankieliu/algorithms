def permute_unique(a):
    a.sort()
    n = len(a)
    used = [False] * n
    res = []
    def bt(path):
        if len(path) == n:
            res.append(path[:])
        for i in range(n):
            if used[i] or (i > 0 and a[i] == a[i-1] and not used[i-1]):
                continue
            used[i] = True
            path.append(a[i])
            bt(path)
            path.pop()
            used[i] = False
    bt([])
    return res
