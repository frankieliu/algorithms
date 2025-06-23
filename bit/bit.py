class BIT:
    def __init__(self, size, a=None):
        if a is not None:
            self.n = len(a)
        else:
            self.n = size
        self.bit = [0] * (self.n + 1)  # 1-based indexing
        if a is not None:
            self.build(a)

    # adds the
    def build(self, a):
        i = 1
        while i <= self.n: 
            self.bit[i] += self.a[i-1]
            ni = i + (i&-i)
            if ni <= self.n:
                self.bit[ni] += self.bit[i]
            i = ni

    # update adds delta to i
    def update(self, i, delta):
        while i <= self.n:
            self.bit[i] += delta
            i += i & -i  # move to parent

    # query gets the prefix sum [1,i]
    def query(self, i):
        res = 0
        while i > 0:
            res += self.bit[i]
            i -= i & -i  # move to previous
        return res

    # gets the sum [l,r]
    def range_query(self, ql, qr):
        return self.query(qr) - self.query(ql - 1)
