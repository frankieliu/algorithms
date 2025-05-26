class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        self.res = None

        def traverse(node):
            if node is None:
                return False
            
            is_cur = node in (p, q)
            is_in_left = traverse(node.left)
            is_in_right = traverse(node.right)
            if (is_cur and (is_in_left or is_in_right)) or (is_in_left and is_in_right):
                self.res = node


            return is_cur or is_in_left or is_in_right
        
        traverse(root)
        return self.res
