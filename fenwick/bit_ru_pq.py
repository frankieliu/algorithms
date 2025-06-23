class BIT_RangeUpdate_PointQuery:
    def __init__(self, size):
        self.n = size
        self.bit = [0] * (self.n + 1)

    def _add(self, i, delta):
        while i <= self.n:
            self.bit[i] += delta
            i += i & -i

    def range_add(self, l, r, delta):
        self._add(l, delta)
        self._add(r + 1, -delta)

    def point_query(self, i):
        res = 0
        while i > 0:
            res += self.bit[i]
            i -= i & -i
        return res
