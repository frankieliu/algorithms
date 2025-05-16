def build(node, start, end):
    if start == end:
        tree[node] = A[start]
        return 
    mid = (start + end) // 2
    build(2*node, start, mid)
    build(2*node+1, mid+1, end)
    tree[node] = tree[2*node] + tree[2*node+1]

def update(node, start, end, idx, val):
    if start == end:
        A[idx] += val
        tree[node] += val
        return
    mid = (start + end) // 2
    # on the left
    if start <= idx and idx <= mid:
        update(2*node, start, mid, idx, val)
    else:
        update(2*node+1, mid+1, end, idx, val)
    tree[node] = tree[2*node] + tree[2*node+1]

def query(node, start, end, l, r):
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
    if r < start or l > end:
        return 0
    # completely within l and r
    if l <= start and end <= r:
        return tree[node]

    # partial overlap
    mid = (start + end) // 2
    p1 = query(2*node, start, mid, l, r)
    p2 = query(2*node+1, mid+1, end, l, r)
    return p1 + p2