def unique_combinations(a, k):
    a.sort()
    n = len(a)
    res = []

    def bt(i, path):
        if len(path) == k:
            res.append(path[:])
            return
        for j in range(i, n):
            if j > i and a[j] == a[j - 1]:
                continue
            path.append(a[j])
            bt(j + 1, path)
            path.pop()

    bt(0, [])
    return res
