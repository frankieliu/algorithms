class BIT:
    def __init__(self, size):
        self.n = size
        self.bit = [0] * (self.n + 1)

    def add(self, i, delta):
        while i <= self.n:
            self.bit[i] += delta
            i += i & -i

    def query(self, i):
        res = 0
        while i > 0:
            res += self.bit[i]
            i -= i & -i
        return res

class BIT_RangeUpdate_RangeQuery:
    def __init__(self, size):
        self.n = size
        self.B1 = BIT(size)
        self.B2 = BIT(size)

    def range_add(self, ql, qr, delta):
        self.B1.add(ql, delta)
        self.B1.add(qr + 1, -delta)
        self.B2.add(ql, delta * (ql - 1))
        self.B2.add(qr + 1, -delta * qr)

    def prefix_sum(self, i):
        return self.B1.query(i) * i - self.B2.query(i)

    def range_sum(self, ql, qr):
        return self.prefix_sum(qr) - self.prefix_sum(ql - 1)
