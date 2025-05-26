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
