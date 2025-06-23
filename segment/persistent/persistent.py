class PersistentSegmentTree:
    """
   
    This is a tree of ranks, for example
    Input has 1,2,3,4,5
    Rank      0,1,2,3,4
    The tree is over the ranks
               0-4
          0-2      3-4
        0-1  2     3 4
        0 1

    In other segment tree, the range is over
    the index within an array, i.e.
    a[l..r] l and r specify a particular leaf in
    the tree.

    HERE, l and r are kept via a history of trees
    since each insertion into the tree creates a
    new root node.  So to answer queries over the
    range l to r, one would have take the differences
    between root @ r+1 an root @ l

    """
    class Node:
        def __init__(self, left=None, right=None, count=0):
            self.left = left
            self.right = right
            self.count = count

    def __init__(self, data):
        self.sorted_data = sorted(set(data))
        # note if there are repeats, then v: {last i}
        # dictionary of the rannk of a particular value
        self.coord = {v: i for i, v in enumerate(self.sorted_data)}
        self.n = len(self.sorted_data)
        self.versions = [self.build(0, self.n - 1)]

        # goes through each value in data
        for val in data:
            # find its rank
            idx = self.coord[val]
            new_root = self.update(self.versions[-1], 0, self.n - 1, idx)
            self.versions.append(new_root)

    def build(self, lb, r):
        """ build just creates all the nodes """
        if lb == r:
            return self.Node()
        mid = (lb + r) // 2
        return self.Node(
            left=self.build(lb, mid),
            right=self.build(mid + 1, r)
        )

    def update(self, node, lb, r, pos):
        """
        pos is value
        1. if at leaf (lb==r), create a new node
           based on the count value of the previous node
        1. otherwise, we need to decide whether the pos
           belongs on the left or right branch
        1. if decide that the pos is on the left branch
           1. create a new node, where left branch is
           a new node, right branch is unchanged, and
           1. count is increased by one
        """
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
        # if we are at the leaf then we have
        # found the answer 
        if lb == r:
            return self.sorted_data[lb]
        mid = (lb + r) // 2

        # versions:
        #  v: version [r+1]
        #  u: version [l]

        #  insertions [l,r] occur btw these versions
        cnt = v.left.count - u.left.count

        # this recursion will eventually land in a leaf node
        if cnt >= k:
            return self.query(u.left, v.left, lb, mid, k)
        else:
            return self.query(u.right, v.right, mid + 1, r, k - cnt)

    def kth_smallest(self, lb, r, k):
        return self.query(self.versions[lb], self.versions[r + 1], 0, self.n - 1, k)
