class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def bstFromPreorder(self, preorder: List[int]) -> Optional[TreeNode]:

        class SF:
            def __init__(self):
                self.lo = None
                self.hi = None
                self.state = 0
                self.root = None
                self.val = None
            def __str__(self):
                return f"{(self.lo,self.hi)} {self.state} {self.val} {self.root}" 

        def bst(lo = -math.inf, hi = math.inf):
            nonlocal idx

            sf = SF()
            sf.lo, sf.hi = -math.inf, math.inf
            st = [sf]

            result = None
            while st:
                # print([str(x) for x in st], result)
                f = st[-1]
                if f.state == 0:
                    if idx == n:
                        result = None
                        st.pop()
                        continue
                    f.val = preorder[idx]        
                    if f.val < f.lo or f.val > f.hi:
                        # importantly don't advance idx
                        result = None
                        st.pop()
                        continue
                    idx += 1
                    f.root = TreeNode(f.val)
                    sf = SF()
                    sf.lo, sf.hi = f.lo, f.val
                    st.append(sf)
                    f.state = 1
                elif f.state == 1:
                    f.root.left = result
                    sf = SF()
                    sf.lo, sf.hi = f.val, f.hi
                    st.append(sf)
                    f.state = 2
                elif f.state == 2:
                    f.root.right = result
                    result = f.root
                    st.pop()
            return result

        idx = 0
        n = len(preorder)
        return bst()