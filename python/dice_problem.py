"""
A six-sided dice is thrown repeatedly with results recorded as int[].
Each time we can only see three sides - top, front, right
(bottom, left, back are hidden).
The visible sides are recorded in the order top, front, right.
So for example [1,2,3] is top = 1, front = 2, and right = 3.
Each side could be a number 1 to 50.

Multiple throws are recorded as [ [1,2,3], [1,3,6], [1,25,45]...]


We need to check that these historical throws data is consistent
with respect to a single six-sided dice.
Write a method consistent(throws: list[list[int]]) → int.
The method returns 1 based index of the first entry in
throws history that is inconsistent with respect to previous ones.
If all the historical data is consistent the method should return 0.

Examples / Test Cases (PLEASE see second page for more)

[[1,2,3], [26,3,2]] Returns 0
First throw 26 is on the bottom, second throw 26 is on top.


[[1,2,3], [4,5,6], [2,3,1], [7,8,9]]  Returns 4
[7,8,9] can’t happen because that would be more than 6 different labels

[[1,2,1], [3,4,5]] Returns 1
First throw two sides are labeled as 1.

Take care and time! to think of additional test cases as necessary. If
you add test cases, document test cases and reasoning what additional
possibilities they test for.
"""

"""
Looking at the possible permutations of TFR:
Top = 1
(1, 2, 3) (1, 3, 5) (1, 5, 4) (1, 4, 2)
Top = 2
(2, 1, 4) (2, 4, 6) (2, 6, 3) (2, 3, 1)
Top = 3
(3, 1, 2) (3, 2, 6) (3, 6, 5) (3, 5, 1)
Top = 4
(4, 1, 5) (4, 5, 6) (4, 6, 2) (4, 2, 1)
Top = 5
(5, 1, 3) (5, 3, 6) (5, 6, 4) (5, 4, 1)
Top = 6
(6, 2, 4) (6, 4, 5) (6, 5, 3) (6, 3, 2)

Given one toss:
# Rule 1
(123) -> (231) -> (312) are permitted and no other {1,2,3}

# Rule 2
(123) and (135) => 2 and 5 are opposite each other
 => opposites cannot be in any tuple

E
(123) and (263) => if two of three are found in another tuple
 => (263) and (326) have to be in this order
"""

def test(d):
    valid = set()      # valid set of (t,f,r) tuples
    visited = set()    # a group of numbers shows up
    opposite = dict() 
    for i, el in enumerate(d):
        t,f,r = el

        key = tuple(sorted([t,f,r]))
        if key in visited:  # make a common key
            if (t,f,r) not in valid:
                return i
        else:
            # Rule 1
            valid.add((t,f,r))
            valid.add((f,r,t))
            valid.add((r,t,f))
        
        # Rule 2
        # here we are trying to if t,x,f
        if opposite(t,f) or opposite(t,r) or opposite(f,r):
            return i
         