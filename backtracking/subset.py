def get_subsets(nums):
    subsets = []

    def backtrack(current_subset, index):

        # Why are we adding the current subset?
        subsets.append(current_subset.copy())

        for i in range(index, len(nums)):
            current_subset.append(nums[i])
            backtrack(current_subset, i+1)
            current_subset.pop()

    backtrack([], 0)
    return subsets

numbers = [1, 2, 3]
all_subsets = get_subsets(numbers)
print(all_subsets)  # Output: [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]