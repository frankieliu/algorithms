from typing import Optional

class TreeNode:
    pass

class Solution:
    def str2tree(self, sss: str) -> Optional[TreeNode]:
        
        if not sss:
            return None

        def solve(ss):
            cur = 0
            sign = 1
            node = None
    
            for s in ss:
                if s == "-":
                    sign = -1
                elif s.isdigit():
                    cur = (cur * 10) + int(s)
                elif s == "(":
                    # Create the main node
                    if not node:
                        node = TreeNode(cur * sign)
                        sign = 1
                        cur = 0
                    child = solve(ss)
                    if not node.left:
                        node.left = child
                    else:
                        node.right = child
                elif s == ")":
                    if not node:
                        node = TreeNode(cur * sign)
                    return node
            if not node:
                node = TreeNode(cur * sign)
            return node

        return solve(iter(sss))