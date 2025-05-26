class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        parrents = {root: None}
        stack = [root] 

        while stack: 
            node = stack.pop()
            if node.left: 
                stack.append(node.left)
                parrents[node.left] = node
            if node.right: 
                stack.append(node.right)
                parrents[node.right] = node

        seen = set()

        while p:
            seen.add(p)
            p = parrents[p]

        while q and not q in seen :
            q = parrents[q]
        
        return q
