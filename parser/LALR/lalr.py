# Created by ChatGPT
# Note: use the one from Deepseek it is based on 
# shift-reduce stack
#
# This one uses too many tricks 
# 1. Incrementing the precedence for associativity
# 1. Recursive calls to implement the stack

import re
from enum import Enum

# Token Type
# keep it short of interview
class TT(Enum):
    mul = r'\*'
    div = r'/' 
    plus = r'\+'
    minus = r'-'
    lparen = r'\('
    rparen = r'\)'
    pow = r'\^'
    num = r'\d+'
    skip = r'\s+'
    eof = None

class Token():
    def __init__(self, type_, value):
        self.type = type_
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, {self.value})"

regex = "|".join(f"(?P<{tt.name}>{tt.value})" for tt in TT if tt is not TT.eof)

def tokenize(text):
    for mo in re.finditer(regex, text):
        kind = TT[mo.lastgroup]
        if kind == TT.skip:
            continue
        value = mo.group()
        if kind == TT.num:
            value = int(value)
        yield Token(kind, value)
    yield Token(TT.eof, None)

""" 
Behavior will only raise StopIteration
on next function
"""
class PeekIterator():
    def __init__(self, iter_):
        self.iter = iter_
        self.has_peek = False
        self.peek_value = None
    
    def peek(self):
        if not self.has_peek:
            try:
                self.has_peek = True
                self.peek_value = next(self.iter)
            except StopIteration:
                self.has_peek = False
                self.peek_value = None
        return self.peek_value

    def __next__(self):
        if self.has_peek:
            self.has_peek = False
            return self.peek_value
        return next(self.iter) 

    def __iter__(self):
        return self

# (precedence, associativity)
# Note the key is not the token type
# BUT the semantic meaning of the token
# For example a '+' can either be a
# Unary Plus or a Binary Plus
# We will call these are 'grammar symbols'
# or 'symbol' for short
PRECEDENCE = {
    TT.pow: (4, 'right'),
    'uminus': (3, 'right'),
    'uplus': (3, 'right'),
    TT.mul: (2, 'left'),
    TT.div: (2, 'left'),
    TT.plus: (1, 'left'),
    TT.minus: (1, 'left'),
    TT.eof: (-1, 'left')
}

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
    
    def next_token(self):
        self.pos += 1
        return next(self.tokens)

    def parse(self):
        return self.parse_expr(0)

    def apply_operator(self, op, lhs, rhs):
        match op.type:
            case TT.plus:
                return lhs + rhs
            case TT.mul:
                return lhs * rhs
            case TT.div:
                return lhs // rhs
            case TT.minus:
                return lhs - rhs
            case TT.pow:
                return lhs ** rhs
        raise Exception(f"apply_operator: {op}")

    """
    expr : expr PLUS expr
         | expr MINUS expr
         | expr MUL expr
         | expr DIV expr
         | expr CARET expr 
         | ( expr )
         | num
    """
    def parse_expr(self, min_prec):
        
        print(f"parse_expr({min_prec}) {self.pos}")
        tt = self.next_token()
        match tt.type:
            case TT.plus:
                # you get the next expression
                lhs = self.parse_expr(PRECEDENCE['uplus'][0])
            case TT.minus:
                lhs = -self.parse_expr(PRECEDENCE['uminus'][0])
            case TT.num:
                lhs = tt.value
            case TT.lparen:
                lhs = self.parse_expr(0)
                if self.tokens.peek().type != TT.rparen:
                    raise Exception(f"Syntax Error at {self.pos}: Expected {TT.rparen} but received {self.token.peek()}")
                self.next_token() # consume rparen
            case _:
                raise Exception(f"Syntax Error at {self.pos}: Unexpected token {tt}")

        print(f"parse_expr({min_prec}) {self.pos} {lhs=}")
        # binary operators
        # Note all unary and () will be 
        # taken by the above
        while True:
            lookahead = self.tokens.peek()
            if lookahead.type not in PRECEDENCE:
                print(f"Not in precedence {lookahead}")
                break
            prec, assoc = PRECEDENCE[lookahead.type]
            
            # shift 
            # smaller precendence than current -> shift
            if prec < min_prec:
                print(f"Breaking out of while: {min_prec=} {lookahead}")
                break
            
            # reduce
            if assoc == "left": # shift here too
                next_min_prec = prec + 1
            else:
                next_min_prec = prec
            op = self.next_token()

            print(f"calling parse_expr 165")
            rhs = self.parse_expr(next_min_prec)
            print(f"reducing {op} {lhs} {rhs}")
            lhs = self.apply_operator(op, lhs, rhs) 
        return lhs


def test_tokenize():
    for tt in tokenize("3+2*4"):
        print(tt)

def test_peak_iterator():
    pi = PeekIterator(iter([1,2,3,4]))
    try:
        while True:
            print(pi.peek())
            print(next(pi))
    except StopIteration:
        pass

def test_parser():
    p = Parser(PeekIterator(tokenize("3+2*4*7+1-2^2^2*-2")))
    res = 3+(2*4*7)+1-(2**2**2)*(-2)
    print(p.parse(), res)

def main():
    # test_tokenize()
    # test_peak_iterator()
    test_parser()

if __name__ == "__main__":
    main()
