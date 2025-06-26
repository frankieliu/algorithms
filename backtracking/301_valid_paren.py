from typing import List

class Solution:

    def removeInvalidParentheses(self, s: str) -> List[str]:

        def helper(ans, reverse):

            if reverse:
                open = ")"
                close= "("
            else:
                open = "("
                close= ")"

            cnt = {open: 0, close: 0}

            removed = 0

            for i in range(len(s)):

                if reverse:
                    c = s[-i-1]
                else:
                    c = s[i]

                # if there is a closing bracket
                # and this is an extra closing bracket
                if c == close and cnt[open] == cnt[close]:

                    tmp = set()

                    # take s initially
                    for string in ans:

                        if reverse:
                            r = range(len(string)-(i+1-removed), len(string))
                        else:
                            # the range only includes
                            # until i
                            r = range(i+1-removed)

                        for j in r:

                            # basically there is an
                            # imbalance, and can
                            # take away any of closing
                            # parenthesis in between?
                            # ( (()) () ) )
                            # can you take away any of these?
                            # ( () () )

                            if string[j] == close:
                                # -----x-----
                                #     ij  
                                tmp.add(string[:j]+string[j+1:])
                                # why is this is a possible sol'n
                                # concept is that any extra ")"

                    ans = tmp
                    removed += 1
                
                elif c in cnt:
                    cnt[c] += 1

            return ans

        ans = {s}
        ans = helper(ans, True)
        ans = helper(ans, False)

        return list(ans)