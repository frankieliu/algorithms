from collections import Counter

def subsets_limited_by_counts(nums):
    result = []
    counter = Counter(nums)
    unique_nums = list(counter.keys())

    def backtrack(index, path):
        print(f"bt({index},{path})")

        if index == len(unique_nums):
            result.append(path[:])
            return

        num = unique_nums[index]
        max_count = counter[num]
        for count in range(0, max_count + 1):
            path.extend([num] * count)
            backtrack(index + 1, path)
            for _ in range(count):
                path.pop()

    backtrack(0, [])
    return result

nums = [1, 1, 2]
print(subsets_limited_by_counts(nums))

"""
how does this backtracking work?
For each unique_nums
generate 0,1,...,count
once all unique_nums are accounted for
add to the result

Example:
{1:2, 2:1}

             [ ] 
              |
     +--------+-----------+
     |        |           |
    []       [1]        [1,1]
     |        |           |
    ++-+    +-+--+      +-+----+
    |  |    |    |      |      |
   [] [2]  [1] [1,2]  [1,1] [1,1,2]

"""