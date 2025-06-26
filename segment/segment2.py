class SegmentTree:
    def __init__(self, a):
        n = len(a)
        self.t = [0] * 4 * n
        self.v = 1
        self.tl, self.tr = 0, n - 1
        self.build(a)

    def build(self, a, v=1, tl=None, tr=None):
        if v == 1:
            tl, tr = self.tl, self.tr
        if tl == tr:
            self.t[v] = a[tl]
        else:
            tm = (tl + tr) // 2
            self.build(a, v * 2, tl, tm)
            self.build(a, v * 2 + 1, tm + 1, tr)
            self.t[v] = self.t[v * 2] + self.t[v * 2 + 1]

    def update(self, pos, val, v=1, tl=None, tr=None):
        if v == 1:
            tl, tr = self.tl, self.tr
        if tl == tr:
            self.t[v] = val
        else:
            tm = (tl + tr) // 2
            if pos <= tm:
                self.update(pos, val, v * 2, tl, tm)
            else:
                self.update(pos, val, v * 2 + 1, tm + 1, tr)
            self.t[v] = self.t[v * 2] + self.t[v * 2 + 1]

    def query(self, l_, r, v=1, tl=None, tr=None):
        if v == 1:
            tl, tr = self.tl, self.tr
        if l_ > tr or r < tl:
            return 0
        if l_ <= tl and tr <= r:
            return self.t[v]
        tm = (tl + tr) // 2
        p1 = self.query(l_, r, v * 2, tl, tm)
        p2 = self.query(l_, r, v * 2 + 1, tm + 1, tr)
        return p1 + p2
