def permute_unique(nums):
    result = []
    nums.sort()  # Sorting is crucial for handling duplicates
    used = [False] * len(nums)

    def backtrack(current_permutation):
        if len(current_permutation) == len(nums):
            result.append(current_permutation.copy())
            return

        for i in range(len(nums)):
            if used[i] or (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]):
                continue
            used[i] = True
            current_permutation.append(nums[i])
            backtrack(current_permutation)
            used[i] = False
            current_permutation.pop()

    backtrack([])
    return result

nums = [1,2,2,3,3,3]
print(permute_unique(nums))