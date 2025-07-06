class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def dfs_recursive(node):
    if node is None:
        return None
    left = node.left
    right = node.right
    return [left, val, right]

def dfs_iterative(root):
    # Define stack frame structure
    class StackFrame:
        __slots__ = ['node', 'state', 'left_result', 'right_result']
        def __init__(self, node):
            self.node = node
            self.state = 0  # 0: before left, 1: after left, 2: after right
            self.left_result = None
            self.right_result = None
    
    if root is None:
        return None
    
    stack = [StackFrame(root)]
    result = None
    
    while stack:
        frame = stack[-1]
        
        if frame.state == 0:
            # Before processing left subtree
            if frame.node.left:
                stack.append(StackFrame(frame.node.left))
            frame.state = 1
            
        elif frame.state == 1:
            # After left subtree is processed
            if frame.node.left:
                frame.left_result = result
                
            # Prepare to process right subtree
            if frame.node.right:
                stack.append(StackFrame(frame.node.right))
                frame.state = 2
            else:
                # No right child, so we're done
                result = [frame.left_result, frame.node.val, None]
                stack.pop()
                
        elif frame.state == 2:
            # After right subtree is processed
            frame.right_result = result
            result = [frame.left_result, frame.node.val, frame.right_result]
            stack.pop()
            
    return result