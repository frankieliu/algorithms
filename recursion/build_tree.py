from collections import deque
from TreeNode import TreeNode

def build_tree(preorder: list) -> TreeNode:
    """Builds a binary tree from a pre-order list (with None for empty nodes)."""
    if not preorder:
        return None
    
    queue = deque(preorder)
    
    def build():
        if not queue:
            return None
        
        val = queue.popleft()
        if val is None:
            return None
        
        node = TreeNode(val)
        node.left = build()
        node.right = build()
        return node
    
    return build()

preorder = [1, 2, 4, None, None, 5, None, None, 3, None, 6, None, None]
root = build_tree(preorder)
print(repr(root))