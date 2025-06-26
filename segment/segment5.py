class SegmentTree:
    def __init__(self, a):
        n = len(a)
        self.t = [0] * 4 * n
        self.v, self.tl, self.tr = 1, 0, n - 1
        self.build(a)

    def build(self, a, v=None, tl=None, tr=None):
        if v is None:
            v, tl, tr = self.v, self.tl, self.tr
        if tl == tr:
            self.t[v] = a[tl]
            return
        tm = (tl + tr) // 2
        self.build(a, v * 2, tl, tm)
        self.build(a, v * 2 + 1, tm + 1, tr)
        self.t[v] = self.t[v * 2] + self.t[v * 2 + 1]

    def update(self, pos, val, v=None, tl=None, tr=None):
        if v is None:
            v, tl, tr = self.v, self.tl, self.tr
        if tl == tr:
            self.t[v] = val
            return
        tm = (tl + tr) // 2
        if pos <= tm:
            self.update(pos, val, v * 2, tl, tm)
        else:
            self.update(pos, val, v * 2 + 1, tm + 1, tr)
        self.t[v] = self.t[v * 2] + self.t[v * 2 + 1]

    def query(self, ql, qr, v=None, tl=None, tr=None):
        if v is None:
            v, tl, tr = self.v, self.tl, self.tr
        if ql > tr or qr < tl:
            return 0
        if ql <= tl and tr <= qr:
            return self.t[v]
        tm = (tl + tr) // 2
        p1 = self.query(ql, qr, v * 2, tl, tm)
        p2 = self.query(ql, qr, v * 2 + 1, tm + 1, tr)
        return p1 + p2
