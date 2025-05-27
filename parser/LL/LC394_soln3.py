# This algo is hard to come up with:
# if first digit, store substr, and accumulate the number
# once a "[", push reps to stack
# once a "]", [substr] multiply by reps (in stack) with substr and add left
# if alpha, just keep add to substr
# substr number []
class Solution:
    def decodeString(self, s: str) -> str:
        stack = []
        reps = 0
        substr = ""

        for c in s:
            print(c, stack, substr)
            if c.isdigit():
                if reps == 0:
                    stack.append(substr)
                    substr = ""
                reps = 10 * reps + int(c)
            elif c == "[":
                stack.append(reps)
                reps = 0
            elif c == "]":
                reps = stack.pop()
                substr = stack.pop() + reps * substr
                reps = 0
            else:
                substr = substr + c

        return substr

def main():
    s = Solution()
    print(s.decodeString("12[ab2[c]]"))

if __name__=="__main__":
    main()