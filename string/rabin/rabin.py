686 Repeated String Match

class Solution(object):
    def repeatedStringMatch(self, A, B):
        def check(index):
            return all(A[(i + index) % len(A)] == x
                       for i, x in enumerate(B))

        # why -1 here?
        q = (len(B) - 1) // len(A) + 1

        p, MOD = 113, 10**9 + 7    # relative prime
        p_inv = pow(p, MOD-2, MOD)
        power = 1

        # Calculate hash of B 
        b_hash = 0
        for x in map(ord, B):
            b_hash += power * x
            b_hash %= MOD
            power = (power * p) % MOD

        # Try to find hash B in hash A
        a_hash = 0
        power = 1
        for i in xrange(len(B)):
            a_hash += power * ord(A[i % len(A)])
            a_hash %= MOD
            power = (power * p) % MOD

        if a_hash == b_hash and check(0): return q

        # Check one above
        power = (power * p_inv) % MOD
        for i in xrange(len(B), (q+1) * len(A)):
            # take off the last one
            a_hash = (a_hash - ord(A[(i - len(B)) % len(A)])) * p_inv
            # insert the newest
            a_hash += power * ord(A[i % len(A)])
            a_hash %= MOD
            # do check if there is a match
            if a_hash == b_hash and check(i - len(B) + 1):