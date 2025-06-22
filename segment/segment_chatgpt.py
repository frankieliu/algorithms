class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (2 * self.n)
        # Build tree by inserting leaf nodes
        for i in range(self.n):
            self.tree[self.n + i] = data[i]
        # Build internal nodes
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.tree[i * 2] + self.tree[i * 2 + 1]

    def update(self, index, value):
        # Set value at position index
        pos = index + self.n
        self.tree[pos] = value
        # Move up and update parents
        while pos > 1:
            pos //= 2
            self.tree[pos] = self.tree[2 * pos] + self.tree[2 * pos + 1]

    def query(self, left, right):
        # Range sum query on interval [left, right)
        result = 0
        left += self.n
        right += self.n
        while left < right:
            if left % 2 == 1:
                result += self.tree[left]
                left += 1
            if right % 2 == 1:
                right -= 1
                result += self.tree[right]
            left //= 2
            right //= 2
        return result
