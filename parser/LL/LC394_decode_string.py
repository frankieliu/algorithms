# To understand this solution
#
# 1. peek() allows one to peek ahead in an iterator
# 1. LL grammar:
#    1. remove left recursion
#    1. make it unambiguous
#       1. str followed by str in naive grammar
# 1. Understand FIRST and FOLLOW
# 1. Understand creating a parse table
# 1. The rest is easy

class peek():
    def __init__(self, iterable):
        self.iter = iter(iterable)
        self.get_next()
    def __iter__(self):
        return self
    def get_next(self):
        try:
            self.next_ = next(self.iter)
            self.stop = False
        except StopIteration:
            self.next_ = None
            self.stop = True
    def peek(self):
        return self.next_
    def __next__(self):
        if self.stop:
            raise StopIteration
        tmp = self.next_
        self.get_next()
        return tmp

class Solution:
    """
    
    # This won't work becaure you can have str followed by str 
    expr := term expr | e
    term := #[ expr ] | str
    str  := letter+ 

    FIRST(expr) = FIRST(term) | e =   # | letter | e
    FIRST(term) =                     # | letter
    FIRST(str) =                      letter 
    FIRST(str') =                     letter | e

    FOLLOW(expr) = ] | $
    FOLLOW(term) = FIRST(expr) | FOLLOW(expr) = # | letter | ] | $
    FOLLOW(str) = FOLLOW(term) = # | letter | ] | $

    |  |   #        | letter |   ]     |
    |--|------------|--------|---------| 
    |E | E->T E     | E->T E |
    |T | T->#[ E ]  | T->S   |
    |S |  S->e      | S->l S |

    # Redo the grammar
    
    Key: str can be followed by a term
    
    expr := str expr' | term expr | e
    expr' := term expr | e
    term := #[ expr ]

    FIRST(expr)   = letter | # | e
    FIRST(expr')  = # | e 
    FIRST(term)   = #
    FOLLOW(expr)  = ], $
    FOLLOW(expr') = ], $
    FOLLOW(term)  = letter | # | ] | $
    FOLLOW(str)   = # | ] | $

          letter     #       ]      $
    E  |  E->S E'  E->TE    E->e   E->e 
    E' |           E'->TE   E'->e  E'->e
    T  |           T->#[E]
    S  |  S->l S

    Remove left recursion
    E := E T
    T := #[ E ]
    ----
    E := T E'
    E':= T E' | e

    """
    def decodeString(self, ss: str) -> str:
        """ 
          letter     #       ]      $
    E  |  E->S E'  E->TE    E->e   E->e 
    E' |           E'->TE   E'->e  E'->e
    T  |           T->#[E]
    S  |  S->l S   S->e     S->e   S->e
        """

        def expr(s):
            res = ""
            if s.peek() is None:
                return ""
            elif s.peek().isalpha():
                return get_string(s) + exprp(s)
            elif s.peek().isdigit():
                return term(s)+expr(s)
            return ""

        def exprp(s):
            res = ""
            if s.peek() is None:
                return ""
            elif s.peek().isdigit():
                t = term(s)
                e = expr(s)
                return t+e
            return ""

        def get_string(s):
            res = ""
            while s.peek() and s.peek().isalpha():
                res += next(s)
            return res

        def get_number(s):
            res = 0
            while s.peek().isdigit():
                res = res*10 + int(next(s))
            return res

        def term(s):
            if s.peek().isdigit():
                num = get_number(s)
                next(s) # get [
                res = expr(s) * num
                next(s) # get ]
            return res 
        
        return expr(peek(ss))