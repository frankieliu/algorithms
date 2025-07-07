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