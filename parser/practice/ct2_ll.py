# This is a simple skeleton to help remember how to do a LL parser

from enum import Enum
import re

class TT(Enum):
    pass

def tokenize(text):
    regex = "|".join(f"(?P<{tt.name}>tt.value)" for tt in TT)
    pass

class Parser:

    def parse(self):
        return self.expr()

    def expr(self):
        """
        E := TE'
        """
        pass

    def expr_p(self, lval):
        """
        E' := + T E' | - T E' | e
        """
        pass

    def term(self):
        """
        T := F T'
        """
        pass

    def term_p(self, lval):
        """
        T' := * P T' | / P T' | e
        """
        pass

    def pow(self):
        """
        P := F | F ^ P
        """
        pass

    def factor(self):
        """
        F = num | - F | + F | ( E )
        """
        pass
