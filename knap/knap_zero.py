"""
There are

$n$ distinct items and a knapsack of capacity
$W$. Each item has 2 attributes, weight (
$w_{i}$) and value ($v_{i}$).

You have to select a subset of items to put into
the knapsack such that the total weight does not
exceed the capacity $W$  and the total value is
maximized.
"""
def kp(wv,cap):
    """ weights[n], values[n], cap """
    """
    dp[i,c] = best value at cap c, having considered 
              up to element i

    dp[i+1, c] = max(dp[i, c-w] + v, dp[i, c])

    You either take the element or not

    We drop i, and just consider getting the i+1 for i
    without clobbering information.
    """
    w = [x[0] for x in wv]
    v = [x[1] for x in wv]
    n = len(w)
    dp = [0]*(cap+1)
    for i in range(n):
        for j in range(cap, w[i]-1, -1):

            dp[j] = max(dp[j-w[i]] + v[i], dp[j])

            # so initially this will fill with v[i]
            # until dp[w[i]], since below this capacity
            # there is no item that will fit

            # Note that because dp[j]
            # required dp[i < j] we need to fill
            # the array from high j to low j
    return dp[cap]