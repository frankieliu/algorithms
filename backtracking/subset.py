def subsets(nums):
    subsets = []

    def backtrack(start, current_subset):

        # Why are we adding the current subset?
        subsets.append(current_subset.copy())

        for i in range(start, len(nums)):
            current_subset.append(nums[i])
            backtrack(i+1, current_subset)
            current_subset.pop()

    backtrack(0, [])
    return subsets

numbers = [1, 2, 3]
all_subsets = subsets(numbers)
print(all_subsets)  # Output: [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]

"""
Note that this is identical to
combination.py, only difference
is that we are outputting every
path.

How does this work
                     []
         [1]         [2]     [3]
     [1,2] [1,3]    [2,3]  
    [1,2,3]
"""