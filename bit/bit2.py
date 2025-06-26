class BIT:
    def __init__(self, a=None):
        self.n = len(a)
        self.t = [0]*(self.n+1)
        self.build(a)

    def build(self, a):
        for i in range(len(a)):
            self.t[i] += a[i]
            j = i + (i&-i)
            if j <= self.n:
                self.t[j] += self.t[i]

    def update(self, i, delta):
        while i <= self.n:
            self.t[i] += delta
            i += (i&-i)

    def query(self, i):
        res = 0
        while i > 0:
            res += self.t[i]
            i -= (i&-i)
        return res

    def range_query(self, ql, qr):
        return self.query(qr) - self.query(ql-1)