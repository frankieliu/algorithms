class BIT:
    def __init__(self, a=None):
        self.n = len(a) 
        self.bit = [0] * (self.n + 1)  # 1-based indexing
        if a is not None:
            self.build(a)

    def build(self, a):
        for i in range(len(a)):
            self.bit[i] += a[i]
            j = i + (i & -i)
            if j <= self.n:
                self.bit[j] += self.bit[i]

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
