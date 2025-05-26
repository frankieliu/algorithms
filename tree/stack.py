class BSTIterator:
    # Algo: Maintain a stack of next elements by adding root 
    # and all left children. On next, pop top of stack, and add the left-
    # children of the right side of the pop-ed node. These are numbers 
    # larger than the node but smaller than the next element of the pop-ed
    # node (cause parent_node.val > any_right_child.val > node.val
    def __init__(self, root: Optional[TreeNode]):
        self.stack = []
        if not root:
            return
        self.stack.append(root)
        while root.left:
            root = root.left
            self.stack.append(root)

    def next(self) -> int:
        ans = self.stack.pop()
        if ans.right:
            node = ans.right
            while node:
                self.stack.append(node)
                node = node.left
        return ans.val
        

    def hasNext(self) -> bool:
        return len(self.stack) > 0 
