class PersistentSegmentTree:
    class Node:
        def __init__(self, left=None, right=None, count=0):
            self.left = left
            self.right = right
            self.count = count

    def __init__(self, data):
        self.sorted_data = sorted(set(data))
        self.coord = {v: i for i, v in enumerate(self.sorted_data)}
        self.n = len(self.sorted_data)
        self.versions = [self.build(0, self.n - 1)]

        for val in data:
            idx = self.coord[val]
            new_root = self.update(self.versions[-1], 0, self.n - 1, idx)
            self.versions.append(new_root)

    def build(self, lb, r):
        if lb == r:
            return self.Node()
        mid = (lb + r) // 2
        return self.Node(
            left=self.build(lb, mid),
            right=self.build(mid + 1, r)
        )

    def update(self, node, lb, r, pos):
        if lb == r:
            return self.Node(count=node.count + 1)
        mid = (lb + r) // 2
        if pos <= mid:
            return self.Node(
                left=self.update(node.left, lb, mid, pos),
                right=node.right,
                count=node.count + 1
            )
        else:
            return self.Node(
                left=node.left,
                right=self.update(node.right, mid + 1, r, pos),
                count=node.count + 1
            )

    def query(self, u, v, lb, r, k):
        if lb == r:
            return self.sorted_data[lb]
        mid = (lb + r) // 2
        cnt = v.left.count - u.left.count
        if cnt >= k:
            return self.query(u.left, v.left, lb, mid, k)
        else:
            return self.query(u.right, v.right, mid + 1, r, k - cnt)

    def kth_smallest(self, lb, r, k):
        return self.query(self.versions[lb], self.versions[r + 1], 0, self.n - 1, k)
