from collections import deque

class Solution:
    def decodeString(self, s: str) -> str:
        stack = []
        for c in s:
            if c != ']':
                stack.append(c)
                continue
            print("Stack]:", stack) 

            # Remove everything inside [word]
            queue = deque()
            while stack[-1] != '[':
                queue.appendleft(stack.pop())
            stack.pop() # pop [
            word = ''.join(queue) 
            print(word)

            # Remove multiplier
            queue = deque()
            while stack and stack[-1].isdigit():
                queue.appendleft(stack.pop())
            new_word = int(''.join(queue)) * word
            stack.extend(new_word)

        return ''.join(stack)