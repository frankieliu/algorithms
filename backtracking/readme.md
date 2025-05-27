[labuladong translated](https://github.com/labuladong/fucking-algorithm/blob/english/think_like_computer/DetailsaboutBacktracking.md)
```
res = []
def backtracking(selection, assignment):
    if end_condition:
        res.append(assignment)
        return
    for x in selection:
        select(x)
        backtracking(assignment + x, new_selection)
        deselect(x) 
```

# Permutation

```
used = [False]*len(selection)
# in permutation you are allowed to select any unused ones
for i in range(len(selection)):
    if used[i]:
        continue

# if there are duplicates then you are allowed to use only if previous has been used
for i in range(len(selection)):
    if used[i] or (i > 0 and selection[i] == selection[i-1] and not used[i-1]):
        continue
```

# Combination

```
# in combination you just use something once, end condition on k
def backtrack(start, assignment):
    if len(assignment) == k:
        res.append(assignment)
        return
    for i in range(start, selection):
        if (i > 0 and selection[i] == selection[i-1]):
            continue
```
Here: 11123, each 1 can be used once, the if prevents using 'future' ones