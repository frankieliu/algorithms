"""
Can you write a ll parser for paren
(()())()

https://stackoverflow.com/questions/16402434/is-this-grammar-for-matched-brackets-ll1

S := TS | e
T := (S)

Each nonterminal that has
more than one production rule,
the director sysmbol sets of the
rules myst be disjoint

DS(S->e) = First(S->e) Union Follow(S-e) = {)}
DS(S->TS) = First(S->TS) = {(}

First(S) = e, (
Follow(S) = ), $
First(T) = )

Parsing Table

     (    )  $
S    TS   e  e
T    (S)
"""

class peek:
     def __init__(self, gen):
          self.gen = gen
          self.get_next()
     def get_next(self):
          try:
               self.next_  = next(self.gen)
               self.stop = False
          except StopIteration:
               self.next_ = None
               self.stop = True
     def peek(self):
          return self.next_
     def __iter__(self):
          return self
     def __next__(self):
          if self.stop:
               raise StopIteration
          tmp = self.next_
          self.get_next()
          return tmp

def t(ss):
     acc = "("
     next(ss)  # (
     acc += s(ss)
     next(ss)  # )
     return acc + ")" 

def s(ss):
     if ss.peek() == '(':
          acc = "<s>"
          acc += t(ss)
          acc += s(ss)
          acc += "</s>"
          return acc
     elif ss.peek() == ')':
          return "<e>"
     elif ss.peek() is None:
          return "<e>"

def main():
     print(s(peek(iter("(()())()"))))

if __name__=="__main__":
     main()