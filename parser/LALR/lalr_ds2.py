import re

def tokenize(s):
    # remove spaces
    groups = {
        'number': r"\d+",
        'paren': r"[()]",
        'operator': r"[+-/*^]",
        'end': r"$"
    }
    expr = "|".join([f"(?P<{k}>{v})" for k,v in groups.items()])
    # print(expr)
    s = re.sub(r"\s+", "", s)
    # print(s)
    res = []
    for mo in re.finditer(expr, s):
        name = mo.lastgroup
        res.append((name, mo[name]))
    return res
# precedence
prec = {
    'end': (0, "left"),
    '+': (1, "left"),
    '-': (1, "left"),
    '*': (2, "left"),
    '/': (2, "left"),
    '^': (3, "right"),
    '-u': (4, "right")
}

def apply(val, op):
    if op[-1] == '-u':
        val.append(-val.pop())
        op.pop()
    elif op[-1] in '+-/*^':
        b,a = val.pop(),val.pop()
        oper = op.pop()
        if oper == '+':
            val.append(a+b)
        elif oper == '-':
            val.append(a-b)
        elif oper == '*':
            val.append(a*b)
        elif oper == '/':
            val.append(a/b)
        elif oper == '^':
            val.append(a**b)
        
def parse(s):
    # value and operator stacks
    val = []
    op = []
    tokens = tokenize(s)
    expect_operand = True
    for type, value in tokens:
        if type == 'number':
            val.append(int(value))
            expect_operand = False
        elif type == 'paren':
            if value == '(':
                op.append(value)
                expect_operand = True
            else:
                while (op and op[-1] != "("):
                    apply(val, op)
                op.pop()
                expect_operand = False
        elif type == 'operator':
            if expect_operand:
                value += "u"
            if value == "+u":
                continue
            while (op and op[-1] != "(" and
                   (prec[value][0] <  prec[op[-1]][0] or
                   (prec[value][0] == prec[op[-1]][0] and
                    prec[value][1] == 'left'))):
                apply(val, op)
            op.append(value)
            expect_operand = True
        elif type == 'end':
            break
        print(val, op)
    while op:
        apply(val, op) 
    return val[-1]

print(parse("2 ^ -3 ^ 2"))