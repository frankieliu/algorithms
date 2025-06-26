# combination problem with repeated elements
# n choose k
def unique_combinations(a,k):
    res = []
    n = len(a)
    def backtrack(i, path):
        # print(i, path)
        if len(path) == k:
            res.append(path[:])
            return
        for j in range(i, n):
            # for aabbbb
            # don't do the second a:
            # - because the current for loop
            #   will already take care of one a 
            # don't do the second b:
            # - for the same reason, because
            #   the first b already takes care of it
            if j > i and a[j] == a[j-1]:
                continue
            path.append(a[j])
            backtrack(j+1, path)
            path.pop()
    backtrack(0, [])
    return res

print(unique_combinations([1,1,2,2,3,3,3,3], 3))
"""
[[1, 1, 2], [1, 1, 3], [1, 2, 2], [1, 2, 3], [1, 3, 3], [2, 2, 3], [2, 3, 3], [3, 3, 3]]
"""
