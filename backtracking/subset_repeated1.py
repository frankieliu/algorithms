from collections import Counter
def subsets_repeated(a):
    res = []
    count = Counter(a)
    keys = list(count.keys())
    n = len(keys)

    def bt(i, path):
        if i == n:
            res.append(path.copy())
            return
        for c in range(count[keys[i]]+1):
            path.extend([keys[i]]*c)
            bt(i+1, path)
            for _ in range(c):
                path.pop()
    bt(0, [])
    return res           

print(subsets_repeated([1,1,2,3,3]))
