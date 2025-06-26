# combination problem
# n choose k
def combine(n,k):
    res = []
    def backtrack(i, path):
        print(i, path)
        if len(path) == k:
            res.append(path[:])
            return
        if i > n:
            return 
        # i is the ith element
        backtrack(i+1, path + [i])
        backtrack(i+1, path) 
    backtrack(1, [])
    return res

print(combine(5,3))
"""
[[1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5], [1, 4, 5], [2, 3, 4], [2, 3, 5], [2, 4, 5], [3, 4, 5]]
"""
