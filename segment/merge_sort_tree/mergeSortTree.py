from bisect import bisect_right

class MergeSortTree:
    def __init__(self, data):
        self.n = len(data)
        self.tree = [[] for _ in range(4 * self.n)]
        self.build(data, 0, 0, self.n - 1)
        self.sorted_data = sorted(set(data))  # For binary search on values

    def build(self, data, node, L, r):
        if L == r:
            self.tree[node] = [data[L]]
        else:
            mid = (L + r) // 2
            self.build(data, 2 * node + 1, L, mid)
            self.build(data, 2 * node + 2, mid + 1, r)
            self.tree[node] = sorted(self.tree[2 * node + 1] + self.tree[2 * node + 2])

    def count_leq(self, node, L, r, ql, qr, x):
        if r < ql or L > qr:
            return 0
        if ql <= L and r <= qr:
            return bisect_right(self.tree[node], x)
        mid = (L + r) // 2
        # over the range count how many element
        # less than or equal to x
        return (self.count_leq(2 * node + 1, L, mid, ql, qr, x) +
                self.count_leq(2 * node + 2, mid + 1, r, ql, qr, x))

    def kth_smallest(self, L, r, k):
        low, high = 0, len(self.sorted_data) - 1
        # do a binary search
        # the range of the search is n therefore
        # log n for the outer loop
        # inner loop is occupied by count_leq 

        while low < high:
            mid = (low + high) // 2

            # self.sorted_data[mid] : the median element
            # count : number of elements that less than the median
            count = self.count_leq(0, 0, self.n - 1, L, r, self.sorted_data[mid])
            
            # count  
            if count < k:
                low = mid + 1
            else:
                high = mid
        return self.sorted_data[low]
