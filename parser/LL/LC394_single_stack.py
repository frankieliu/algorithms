class Solution:

    def decodeString(self, s: str) -> str:

        def get_string(i):
            res = "" 
            while i < len(s) and s[i].isalpha():
                res += s[i]
                i+=1
            return i, res

        def get_number(j):
            i = j
            while j < len(s) and s[j].isdigit():
                j+=1
            return j, int(s[i:j])

        stack = []
        i = 0
        while i < len(s):
            c = s[i]
            if c.isalpha():
                i, res = get_string()
                stack.append(res)
            elif c.isdigit():
                i, res = get_number()
                stack.append(res)
            elif c == "[":
                stack.append('[')
                i += 1
            elif c == "]":
                j = len(stack) - 1
                while stack[j] != "[":
                    j -= 1
                word = "".join(stack[j+1:len(stack)])
                for _ in range(len(stack)-1,j-1,-1):  # also pop [
                    stack.pop()
                reps = stack.pop()
                stack.append(reps*word)
                i += 1
        return ''.join(stack)