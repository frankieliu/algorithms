# define the grammar
'''
sum = sum + prod | sum - prop | prod
prod = prod * factor | prod / factor | factor
factor = (sum) | number
'''
# make it ll(1) - no left recursion
'''
sum = prod sum_p
sum_p = e | + prod sump | - prod sump
prod = factor prod_p
prod_p = e | * factor prod_p | / factor prod_p
factor = (sum) | number
'''
import re
from enum import Enum

# token type
class TT(Enum):
    num = r'\d+'
    plus = r'\+'
    minus = r'-'
    mul = r'\*'
    div = r'/'
    lparen = r'\('
    rparen = r'\)'
    skip = r'\s+'
    eof = None

regex = '|'.join(f'(?P<{tt.name}>{tt.value})' for tt in TT if tt is not TT.eof)

class Token:
    def __init__(self, type_, value=None):
        self.type = type_
        self.value = value
    def __repr__(self):
        return f'Token({self.type}, {self.value})'

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
E := T E'
E' := + T E' | - T E' | e
T := F T'
T' := * F T' | / F T' | e
F := (E) | Num

Updated grammar with unary and power

Expr := Term Expr'
Expr' := + Term Expr' | - Term Expr' | e
Term := Power Term'
Term' := * Power Term' | / Power Term' | e
Power := Unary ^ Power | Unary  <- note this is not left recursive!
Unary := - Unary | + Unary | Factor
Factor := ( Expr ) | Number

"""

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.current_token = next(self.tokens)
    
    def eat(self, token_type):
        if self.current_token.type != token_type:
            raise Exception(f'Unexpected token: {self.current_token}, expected: {token_type}')
        self.current_token = next(self.tokens)
    
    def parse(self):
        return self.expr()
    
    def expr(self):
        result = self.term()
        return self.expr_prime(result)
    
    def expr_prime(self, prev):
        match self.current_token.type:
            case TT.plus:
                self.eat(TT.plus)
                result = prev + self.term()
                return self.expr_prime(result)
            case TT.minus:
                self.eat(TT.minus)
                result = prev - self.term()
                return self.expr_prime(result)
        return prev

    def term(self):
        result = self.factor()
        return self.term_prime(result)
    
    def term_prime(self, prev):
        match self.current_token.type:
            case TT.mul:
                self.eat(TT.mul)
                result = prev * self.factor()
                return self.term_prime(result)
            case TT.div:
                self.eat(TT.div)
                result = prev // self.factor()
                return self.term_prime(result)
        return prev

    def factor(self):
        match self.current_token.type:
            case TT.lparen:
                self.eat(TT.lparen)
                result = self.expr()
                self.eat(TT.rparen)
                return result
            case TT.num:
                result = self.current_token.value
                self.eat(TT.num)
                return result
        raise Exception(f'Unexpected factor: {self.current_token}') 

def main():
    parser = Parser(tokenize('3+2*4-2'))
    print(parser.parse())

if __name__=="__main__":
    main()
