
def kp(weights, values, counts, W):

    # go through all the elements
    for i in range(n):  # for each item
        w, v, c = weights[i], values[i], counts[i]

        # residue class just depends on the
        # current w
        for r in range(w):  # group by mod class
            dq = deque()

            max_k = (W - r) // w  # how many steps in this group
            # this is going from right to left
            # this is ok because we are appending the older elements
            # into the deque, so they don't get clobbered when
            # writing 
            for k in range(max_k + 1):
                "idx =          r, w+r,       2*w+r,       3*w+r, ..."
                "val =  dp[0] - 0, dp[w+r]-v, dp[2w+r]-2v, dp[3w+r]-3v"
                idx = k * w + r
                val = dp[idx] - k * v  # g(k)

                """ 
                at dp[3w+r] - 3v:
                what values might be in the deqeue?
                (0, dp[r])
                (1, dp[w+r]-v)
                (2. dp[2w+r]-2v)
                (3, dp[3w+r]-3v)

                0w   1w   2w   3w      
                 0  -1v  -2v  -3v

                In general the value at k=3
                should be the max of

                max(
                dp[3]
                dp[2] + v
                dp[1] + 2v
                dp[0] + 3v
                )

                but this is the same as
                max(
                dp[3] - 3v
                dp[2] - 2v
                dp[1] - 1v
                dp[0] - 0v
                ) + 3v

                writing more explicitly
                max(
                dp[j],
                dp[j-1w] + v
                dp[j-2w] + 2v
                ...
                dp[j-kw] + kv
                )

                max(
                dp[j]     - kv    
                dp[j-1w]  - (k-1)v
                dp[j-2w]  - (k-2)v
                dp[j-kw]  - (k-w)v
                ) + kv
                We take the max of these
                """
                # maintain decreasing deque
                # remove elements that are smaller than the current element
                while dq and dq[-1][1] <= val:
                    dq.pop()
                dq.append((k, val))

                # pop out-of-window elements (older than count limit)
                # for the current k, only consider elements which have
                # counts from 0 to c
                while dq and dq[0][0] < k - c:
                    dq.popleft()

                # max g(k) + k*v
                dp[idx] = dq[0][1] + k * v
