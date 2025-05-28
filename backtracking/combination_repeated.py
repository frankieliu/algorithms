def unique_combinations(nums, k):
    result = []
    nums.sort()  # Sort to handle duplicates

    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        for i in range(start, len(nums)):
            # Skip duplicates
            if i > start and nums[i] == nums[i - 1]:
                continue
            path.append(nums[i])
            backtrack(i + 1, path)  # move forward to avoid reusing elements
            path.pop()

    backtrack(0, [])
    return result

print(unique_combinations([1,1,2],2))
"""
Same as before:

112
                   []
     [1]           [x] [2]    Note: don't reuse 1 as single element
[1,1]    [1,2]
[1,1,2]
"""