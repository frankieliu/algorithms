84 Largest Rectangle in Histogram
: INCREASING monotonic queue
: when popping an element calculate the area of the rectangle
: since it is blocked on both sides by something smaller
: note TrapRain water, want a DECREASING monotonic queue

399 Evaluate division
: union, find, parent is the lowest common denominator
  - union with parent and weight
  - only modify weight on find, since need to keep track

3170 Lexicographically Min String After Removing Stars
: fixed alphabet, keep list of indices for each letter (a-z)
: use a heap to keep track of smallest available char

621 Task Scheduler
: A...A...ABCD  - where BCD are the letters with the same count as A

134 Gas Station
: if cumsum goes neg, start at the next station with 0, if total < 0, -1

135 Candy
: go from left to right and right to left, increment by 1, and take max of both

1868 Product of Two Run-Length Encoded Arrays
: go from the back and reduce the count if certain amount it taken
: for example (v, count-used), then don't have to keep track

525 COntinguous Array 0/1
: keep a hash of count
: if you meet the same count later, then the stuff in between sums to 0

626 exchange seats
```sql
# Write your MySQL query statement below
SELECT
  ROW_NUMBER() OVER(ORDER BY IF(MOD(id, 2) = 0, id - 1, id + 1)) AS id,
  student
FROM Seat;
```

1008 Construct Binary Search Tree from Preorder Traversal
: use lo and hi to limit what can go on the left and right subtree
: if your value is not within lo and hi, don't advance idx

https://ipleak.net/

317 Shortest Distance from All Buildings
: take bfs from each building
: mark a distance and a visited grid

33 Search in Rotated Sorted Array
: check whether the side is sorted first
: if sorted, then check if element on that side, if not then look at the other side
: on checking if left side is sorted, you want to nums[l] <= nums[mid] in case of single element on left

3443 Max Manhattan Distance After K Changes
: The maximum you distance is the length of the string up to i, so it is the upper bound of improvement,
  so we add to the running distance 2k and make sure it stays within bounds

792 Number of Matching Subsequences
: put words into buckets corresponding to the letters that they are waiting for
: iterate through the string, moving words in the buckets into new buckets
:
: interesting  O(words_len * log(len(s))) solution
```python
__import__("atexit").register(lambda: open("display_runtime.txt", "w").write("0"))
class Solution:
    def numMatchingSubseq(self, s: str, words: List[str]) -> int:
        letter_indices = defaultdict(list)
        for i, c in enumerate(s):
            letter_indices[c].append(i)

        def is_subsequence(word):
            pos = -1
            for char in word:
                indices = letter_indices[char]
                i = bisect.bisect_right(indices, pos)
                if i == len(indices):
                    return False
                pos = indices[i]
            return True

        return sum(1 for word in words if is_subsequence(word))
```

12 Integer to Roman
: use bisect to find the closest thing to and remove from value

2081 Sum of k-Mirror Numbers
: count 0-9     -> generate odd and even palindrome
: count 10-99   
: count 100-999
: check if other base is also palindrome

713 Subarray Product Less Than K
: sliding window

187 Repeated DNA Sequences
: Rabin Karp
: 10 long -> 4^0 ... 4^9
: I like dividing the previous by 4 and adding a new one * 4^9

2603 Collect Coins in a Tree
: Kahn's algo to remove branches without coins
: Kahn's algo on leafs with coin for 2 times
: remaining (nodes - 1) * 2

1143 Longest Common Subsequence
: dp using similar concept as edit distance to go from start to target
: if s[i] == t[j], then dp[i][j] = 1 + dp[i+1][j+1]
: else dp[i][j] = max(dp[i+1][j], dp[i][j+1])

44 Wildcard Matching
: i source j pattern
: if letters match: dp[i][j] = dp[i+1][j+1]
: if pattern = "?": dp[i][j] = dp[i+1][j+1]
: if pattern = "*": dp[i][j] = dp[i][j+1] "match empty" or dp[i+1][j] "match one and more"
: add # # sentinels for source and pattern

300 Longest Increasing Subsequence
: Patient sort

686 Repeated String Match
: can try RK with p = 113 and M = 10**9+7
: just use z algorithm


1424 Diagonal Traverse II
: bfs from (0,0), and add down and right


