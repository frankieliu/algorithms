""" create permutation with repeats """


def permute_unique(nums):
    """ nums: input an array """

    result = []
    nums.sort()  # Sorting is crucial for handling duplicates
    used = [False] * len(nums)

    def backtrack(current_permutation):
        if len(current_permutation) == len(nums):
            result.append(current_permutation.copy())
            return

        for i in range(len(nums)):

            # used[i] could be useful in normal permutation
            # it prevents future permutation to add the same
            # [i], instead of checking done in permutation.py

            # This check is similar to combinations
            #
            # i > 0 and nums[i] == nums[i-1]
            #
            # aabbb: second a b don't have to be considered, since
            #        first a takes care of having an a, but only use
            #        this condition if the first a is NOT used
            #        i.e. the second a can be used only if the previous
            #        a is used!!!

            if used[i] or (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]):
                continue
            used[i] = True
            current_permutation.append(nums[i])
            backtrack(current_permutation)
            used[i] = False
            current_permutation.pop()

    backtrack([])
    return result


def main():
    """ -- """
    nums = [1, 2, 2, 3]
    print(permute_unique(nums))


if __name__ == "__main__":
    main()
