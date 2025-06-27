The key intuition is that the weight
of the repeated element only needs to
consider positions specific positions in dp[j].

In particular to figure out best dp[j],
I just need to look for the max
[dp[j-kw] - kv] for k in (0,K),

w      v     K
weight value number

For thinking:
Going from left to right is like the 
traditional max in sliding window problem.

But in following knapsack 0/1 problem,
if we move from right to left, then we
can write the answer for dp[j] and not
worry about clobbering the solution.

Note that the looping is a bit more
intricate:

for y in range(0, -w, -1):
    for x in range(W//w, -1, -1):
        dp[x*w+y] = max(sliding_window)
