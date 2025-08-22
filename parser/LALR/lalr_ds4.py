prec = {
    "+": (1, "left"),
    "-": (1, "left"),
    "*": (2, "left"),
    "/": (2, "left"),
    "^": (3, "right"),
    "u-": (4, "right")
}

def tokenize(expr):
    i = 0
    res = []
    while i < len(expr):
        ch = expr[i] 
        if ch.isdigit():
            num = 0
            while ch.isdigit() and i < len(expr):
                num = num*10 + int(ch)
                i += 1
            res.append(("num", num))
        elif ch.isspace():
            i+=1
        elif ch in "+-*/^()":
            res.append(("op",ch))
            i+=1
    return res
            
def apply_ext(vl, op):
    pass

def parse(expr):
    vl, opl = [],[]
    tokens = tokenize(expr)
    expectOperand = True
    for op, val in tokens:
        if op == 'num':
            vl.append(val)
            expectOperand = False