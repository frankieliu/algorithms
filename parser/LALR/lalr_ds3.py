def tokenize(s):
    i = 0
    res = []
    while i < len(s):
        ch = s[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit():
            num = 0
            while i < len(s) and s[i].isdigit():
                num = num * 10 + int(s[i])
                i += 1
            res.append(("num", num))
        elif ch in "+-*/^()":
            res.append(("op", ch))
            i += 1
    return res


def apply_ext(val, op):
    opr = op.pop()
    if opr == "u-":
        val.append(-val.pop())
    elif opr in "+-*/^":
        b = val.pop()
        a = val.pop()
        # print(f"{a=} {opr=} {b=}")
        if opr == "+":
            a += b
        elif opr == "-":
            a -= b
        elif opr == "*":
            a *= b
        elif opr == "/":
            a /= b
        elif opr == "^":
            a = a**b
        val.append(a)


prec = {
    "+": (1, "left"),
    "-": (1, "left"),
    "*": (2, "left"),
    "/": (2, "left"),
    "^": (3, "right"),
    "u-": (4, "right"),
}


def parse(expr):
    vl, op = [], []

    def apply():
        apply_ext(vl, op)

    tokens = tokenize(expr)
    expect_operand = True
    for type, val in tokens:
        if type == "num":
            vl.append(val)
            expect_operand = False
        elif val == "(":
            op.append(val)
            expect_operand = True
        elif val == ")":
            while op and op[-1] != "(":
                apply()
            op.pop()
        elif type == "op":
            if expect_operand:
                if val == "+":
                    continue
                elif val == "-":
                    val = "u-"
            while (
                op
                and op[-1] != "("
                and (
                    prec[val][0] < prec[op[-1]][0]
                    or (prec[val][0] == prec[op[-1]][0] and prec[val][1] == "left")
                )
            ):
                apply()
            op.append(val)
            expect_operand = True
        # print(vl, op)
    while op:
        apply()
    return vl[-1]


print(parse("3 + -5 * 2"))
