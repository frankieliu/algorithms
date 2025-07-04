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
