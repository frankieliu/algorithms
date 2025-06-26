def permute(arr):
    result = []
    n = len(arr)

    # here index is just counting the size of the current_permutation
    def backtrack(current_permutation, index):
        if index == n:
            # Make a copy of the permutation
            result.append(current_permutation[:])
            return

        for i in range(n):
            if arr[i] not in current_permutation:
                current_permutation.append(arr[i])
                backtrack(current_permutation, index+1)
                current_permutation.pop()  # Backtrack

    backtrack([], 0)
    return result


# Example usage
my_array = [1, 2, 3]
permutations = permute(my_array)
print(permutations)
