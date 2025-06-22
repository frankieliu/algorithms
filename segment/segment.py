class SegmentTree:
    def __init__(self, a):
        n = len(a)
        self.t = [0] * 4 * n
        self.tl, self.tr = 0, n - 1
        self.v = 1
        self.build(a)

    # adding an array to the tree
    # at root it should be called with
    # v = 1, tl = 0, tr = n-1
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

    def update(self, pos, new_val, v=1, tl=None, tr=None):
        if v == 1:
            tl, tr = self.tl, self.tr
        if tl == tr:
            self.t[v] = new_val
        else:
            tm = (tl + tr) // 2
            if pos <= tm:
                self.update(pos, new_val, v * 2, tl, tm)
            else:
                self.update(pos, new_val, v * 2 + 1, tm + 1, tr)
            self.t[v] = self.t[v * 2] + self.t[v * 2 + 1]

    def sum(self, ll, r, v=1, tl=None, tr=None):
        if v == 1:
            tl, tr = self.tl, self.tr
        if ll > r:
            return 0
        if ll == tl and r == tr:
            return self.t[v]
        tm = (tl + tr) // 2
        p1 = self.sum(ll, min(tm, r), v * 2, tl, tm)
        p2 = self.sum(max(ll, tm + 1), r, v * 2 + 1, tm + 1, tr)
        return p1 + p2

    def query(self, ll, r, v=1, tl=None, tr=None):
        """
        start and end are the nodes range
        l and r are the query ranges

        3 conditions:

        1. inside -> return the node value

                l         r
                [     ]

        1. partial -> recurse down the tree

                l         r
            [     ]

        1. outside -> return 0
                l         r
                            [    ]
        """
        if v == 1:
            tl, tr = self.tl, self.tr
        if r < tl or ll > tr:
            return 0

        # completely within l and r
        if ll <= tl and tr <= r:
            return self.t[v]

        # partial overlap
        tm = (tl + tr) // 2
        p1 = self.query(ll, r, v * 2, tl, tm)
        p2 = self.query(ll, r, v * 2 + 1, tm + 1, tr)
        return p1 + p2
