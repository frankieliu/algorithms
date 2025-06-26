# Fractional cascade
# 
# When merging of child nodes
# Keep track of individual indexes in child nodes
# e.g.
# 2 4 5 6 6 9 11  (after merging)
# 0 0 1 2 2 2  3  (positions within left tree - l_index)
# 0 1 1 2 3 3  3  (posiitons within right tree - r_index)
# L - L L - -  L  (just a helper to illustrate which 
# - R - - R R  -   branch a particular number belongs 
#
# Idea is to do one binary search
# at the root node in log n time as opposed
# to (log n)^2
# 
# so instead of descending into child nodes
# it may be sufficient to look
# at the ordering from left_idx and right_idx

from bisect import bisect_right

class FCNode:
    def __init__(self):
        self.data = []
        self.left_index = []
        self.right_index = []

class FractionalCascadingMergeSortTree:
    def __init__(self, data):
        self.n = len(data)
        self.data = data
        self.tree = [FCNode() for _ in range(4 * self.n)]
        self.build(0, 0, self.n - 1)

    def build(self, node, lb, r):
        if lb == r:
            self.tree[node].data = [self.data[lb]]
            return

        mid = (lb + r) // 2
        self.build(2 * node + 1, lb, mid)
        self.build(2 * node + 2, mid + 1, r)

        left = self.tree[2 * node + 1].data
        right = self.tree[2 * node + 2].data
        merged = []
        l_idx, r_idx = [], []

        # merge step
        # nice way of merging...
        i = j = 0  # i and j idx into left and right branches, resp
        while i < len(left) or j < len(right):
            if j == len(right) or (i < len(left) and left[i] <= right[j]):
                merged.append(left[i])
                l_idx.append(i) 
                r_idx.append(j)
                i += 1
            else:
                merged.append(right[j])
                l_idx.append(i)
                r_idx.append(j)
                j += 1

        self.tree[node].data = merged
        self.tree[node].left_index = l_idx
        self.tree[node].right_index = r_idx

    def count_leq(self, node, lb, r, ql, qr, x, idx=None):
        if r < ql or lb > qr:
            return 0

        # if completely enclosed then just bisect
        if ql <= lb and r <= qr:
            if idx is None:
                idx = bisect_right(self.tree[node].data, x)
            return idx

        if idx is None:
            idx = bisect_right(self.tree[node].data, x)

        # idx points to the element smallest element bigger or equal to x
        # left_index[idx-1] -> why do we point -1 and the add a 1?
        left_idx = self.tree[node].left_index[idx - 1] + 1 if idx > 0 else 0
        right_idx = self.tree[node].right_index[idx - 1] + 1 if idx > 0 else 0
        
        mid = (lb + r) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2

        return (
            self.count_leq(left_child, lb, mid,     ql, qr, x, left_idx) +
            self.count_leq(right_child, mid + 1, r, ql, qr, x, right_idx)
        )

    def kth_smallest(self, lb, r, k):
        all_vals = sorted(set(self.data))
        lo, hi = 0, len(all_vals) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            count = self.count_leq(0, 0, self.n - 1, lb, r, all_vals[mid])
            if count < k:
                lo = mid + 1
            else:
                hi = mid
        return all_vals[lo]
