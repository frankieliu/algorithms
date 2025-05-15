def tokenize(text):
    l = len(text)
    i = 0
    while i < l:
        match text[i]:
            case ch if ch in '+-/*^()':
                i += 1
                yield ('OP', ch)
            case ch if ch.isspace():
                i += 1
            case ch if ch.isdigit():
                res = "" 
                while i < l and text[i].isdigit():
                    res += text[i]
                    i += 1
                yield ('NUM', int(res))
            case _:
                raise ValueError(f"tokenize: {text[i]=}")
        
            
        if text[i].isdigit():
            res = ""
            while i < l and text[i].isdigit():
                res += text[i]
                i += 1
            yield ('NUM', int(res))

PRECEDENCE = {
    "+": (1, "left"),
    "-": (1, "left"),
    "*": (2, "left"),
    "/": (2, "left"),
    "^": (3, "right"),
    "u+": (4, "right"),
    "u-": (4, "right")
}

def apply_op(values, operators):
    op = operators.pop()
    match op:
        case 'u+':
            pass
        case 'u-':
            res = values.pop()
            values.append(-res)
        case opp if opp in '*/+-^':
            b = values.pop()
            a = values.pop()
            match opp:
                case '*':
                    values.append(a*b)
                case '/':
                    values.append(a//b)
                case '+':
                    values.append(a+b)
                case '-':
                    values.append(a-b)
                case '^':
                    values.append(a**b)
        case _:
            raise ValueError(f"apply: {op=}")

def parse(tokens):
    values = []
    operators = []    
    def apply():
        apply_op(values, operators)
    expect_operand = True
    for token in tokens:
        # print(f"{token} {values} {operators}")
        symbol, value = token
        if symbol == 'NUM':
            values.append(value)
            expect_operand = False
            continue
        if symbol == 'OP':
            if value == "(":
                operators.append(value)
                expect_operand = True
                continue
            if value == ")":
                while operators[-1] != "(":
                    apply()
                operators.pop()
                expect_operand = False
                continue
            if expect_operand and value in "+-":
                operators.append("u"+value)
                expect_operand = True
                continue
            while operators and operators[-1] != '(':
                prev_prec = PRECEDENCE[operators[-1]][0]
                cur_prec = PRECEDENCE[value][0]
                cur_assoc = PRECEDENCE[value][1]
                if ((cur_prec < prev_prec) or 
                    (cur_prec == prev_prec and cur_assoc == 'left')):
                    apply()
                else:
                    break

            operators.append(value)
            expect_operand = True

    while operators:
        apply() 
    return values[0]     

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
        result = parse(tokenize(expr))
        print(f"{expr:15} = {result}")