def combine(n, k):
    results = []
    
    def backtrack(start, combination):
        if len(combination) == k:
            results.append(combination.copy())
            return
        
        for i in range(start, n + 1):
            combination.append(i)
            backtrack(i + 1, combination)
            combination.pop()
    
    backtrack(1, [])
    return results

n = 5
k = 3
combinations = combine(n, k)
print(combinations)
# Expected Output: [[1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5], [1, 4, 5], [2, 3, 4], [2, 3, 5], [2, 4, 5], [3, 4, 5]]
"""
                                         []
               [1]                      [2]      [3] [4] [5]
   [1,2] [1,3]   [1,4]   [1,5]    [2,3]
 [1,2,3] [1,3,4] [1,4,5]
 [1,2,4] [1,3,5]  
 [1,2,5]

1. Basically fill in one element at a time
1. Don't reuse previous elements
"""