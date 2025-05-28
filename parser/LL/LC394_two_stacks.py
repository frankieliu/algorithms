# Similar to single stack solution
# 1. keep the count in a separate count_stack
# 1. string_stack keeps track of prefix strings (decoded strings, maybe "")
# 1. [ : append the current_string to string_stack 
# 1. ] : current_string = prefix + reps * current_string
# 1. This allows the current_string to be updated

class Solution:
    def decodeString(self, s: str):

        def get_number(i):
            j = i
            while s[j].isdigit():
                j+=1
            return j, int(s[i:j])

        def get_string(i):
            j = i
            while j < len(s) and s[j].isalpha():
                j+=1
            return j, s[i:j]

        count_stack = []
        string_stack = []
        current_string = ""
        i = 0
        while i < len(s):
            c = s[i]
            if c.isdigit():
                i, res = get_number(i)
                count_stack.append(res)
            elif c.isalpha():
                i, res = get_string(i)
                current_string += res
            elif c == "[":
                string_stack.append(current_string)
                current_string = ""
                i += 1
            elif c == "]":
                current_string = string_stack.pop() + count_stack.pop() * current_string
                i += 1
        return current_string
