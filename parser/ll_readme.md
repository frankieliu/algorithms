# calculator

expr -> expr + expr | expr - expr | expr * expr | expr / expr | expr ^ expr | - expr | + expr | (expr)

# try to remove left recursion

expr -> expr + expr | n
expr -> n pexpr
pexpr -> + n pexpr | e

a -> a b | c

Transformation:

a -> c a'
a' -> b a' | e

# precedence