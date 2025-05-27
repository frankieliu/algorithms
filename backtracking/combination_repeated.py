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