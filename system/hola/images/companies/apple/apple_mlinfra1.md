# Apple interview

# Your previous Plain Text content is preserved below:


# everyday we get a number of tree logs of different length in our distribution center. We have to cut and distribute the wood to K factories. You may cut any of the log into any number of pieces or perform no cuts at all. Your goal is to distribute this wood to the  k factories. all of the pieces should be of same length. Return the maximum possible length for a piece of log you can obtain to distribute into K factories, or 0 if you cannot

# k factories
# logs
# 10, 5, 4, 8  (log length)
# 3 factories
# length of log to cut so I can distribute as much
# gcd (10, 5, 4, 8)


# Trees - 7, 6, 8 
# factories = 5
# ans = 3

# 7 - (4,3) (3,3,1) (2,5)
# 6 - 3,3 - (2,4) .. (1,5)
# 8 - (4,4) .. 

"""
 k: number
  logs: huge array"
"""

"""
can_satisfy:
parallize and gather
[0:1e6], [1e6+1,..], 
gather results (aggregation)

DC_i: factories_i, logs_i

DC_j: factories_j, logs_j

Globally, one cut length
"""
"""
def best_length(logs, k):
    "k: num factories"
    "logs : array of log length"
    def can_satisfy(m):
        res = 0
        for x in logs: 
            res += x//m
        return res >= k
    if sum(logs)<k:
        return 0
    if len(logs) == 0:
        return 0
    lo, hi = 1, max(logs)
    ans = 0 
    while lo <= hi:
        mid = lo + (hi-lo)//2
        if can_satisfy(mid):
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ans

logs = [7,6,8]
k = 5
print(best_length(logs, 5))

# Trees - 3,2,4,5
# factories - 15
logs = [3,2,4,5]
k = 15
print(best_length(logs, k))

logs = [3, 3, 3]
k = 9

logs = [3, 3, 3]
k = 3
print(best_length(logs, k))

logs = [1,100]
k = 1
print(best_length(logs, k))

# O(1) space
# O(log(max(logs))*len(logs)) time
"""





# Implement a code to do random oversampling 
import random
def sample(a, n_samples):
    out = []
    visited = set()    
    if len(a) < n_samples:
        raise ValueError
    for i in range(n_samples):
        index = random.randint(0,len(a)-1)
        if index in visited:
            continue
        visited.add(index)
        out.append(a[index])
    return out

print(sample([1,2,3,4], 3)