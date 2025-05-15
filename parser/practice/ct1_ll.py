# coding test
# tokenizer
from enum import Enum

# token type

class TT(Enum):
    plus = r'\+'
    minus = r'-'
    mul = r'\*'
    div = r'/'
    lparen = r'\('
    rparen = r'\)'
    pow = r'\^'
    eof = r'\$'
    skip = r'\s+'
    num = r'\d+'

regex = '|'.join(f"(?P<{tt.name}>{tt.value})" for tt in TT)

class Token:
    def __init__(self, type_, value=None):
        self.type = type_
        self.value = value
    def __repr__(self):
        return f"Token({self.type}, {self.value})"

import re
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


# parser
class LLParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.cur_token = next(self.tokens)

    def eat(self, tt):
        if self.cur_token.type != tt:
            raise Exception(f"Parser eat expected {tt} got {self.cur_token.type}")
        self.cur_token = next(self.tokens)

    def parse(self):
        return self.expr()

    def expr(self):
        """E := TE'"""
        res = self.term()
        return self.expr_p(res)
    
    def expr_p(self, prev):
        """E' := + T E' | - T E' | e""" 
        match self.cur_token.type:
            case TT.plus:
                self.eat(TT.plus)
                res = self.term()
                return self.expr_p(prev + res)
            case TT.minus:
                self.eat(TT.minus)
                res = self.term()
                return self.expr_p(prev - res)
        return prev

    def term(self):
        res = self.pow()
        return self.term_p(res)

    def term_p(self, prev):
        match self.cur_token.type:
            case TT.mul:
                self.eat(TT.mul)
                res = self.pow()
                return self.term_p(prev * res)
            case TT.div:
                self.eat(TT.div)
                res = self.pow()
                return self.term_p(prev // res)
        return prev

    def pow(self):
        """
        P := F ^ P | F
        """
        res = self.factor()
        if self.cur_token.type == TT.pow:
            self.eat(TT.pow)
            return res ** self.pow()
        return res
    
    def factor(self):
        """
        F := num | ( expr ) | - F | + F | EOF
        """
        print(f"factor: {self.cur_token}")
        match self.cur_token.type:
            case TT.lparen:
                self.eat(TT.lparen)
                res = self.expr()
                self.eat(TT.rparen)
                return res
            case TT.plus:
                self.eat(TT.plus)
                return self.factor()
            case TT.minus:
                self.eat(TT.minus)
                return -self.factor()
            case TT.num:
                res = self.cur_token.value
                self.eat(TT.num)
                return res
        raise Exception(f"factor: {self.cur_token}")


        

# Test cases
if __name__ == "__main__":
    test_cases = [
        "3 + -5 * 2",      # -7
        "-3 * 5 + 2",      # -13
        "2 ^ -3 ^ 2",      # 2^(-(3^2)) ≈ 0.00195
        "-(10 - 3) - 2",   # -9
        "5 + + + 3",       # 8
        "5 + - - 3",       # 8
        "- (2 + 3) * 4",   # -20
        "3*-2",            # -6
        "3+-2",            # 1
    ]

    for expr in test_cases:
        p = LLParser(tokenize(expr))
        result = p.parse()
        print(f"{expr:15} = {result}")