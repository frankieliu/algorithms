from collections.abc import Generator
from typing import Union

def tokenize(text: str) -> Generator[tuple[str, Union[str, int]]]:
    l = len(text) 
    i = 0
    while i < l:
        ch = text[i]
        if ch in '+-/*^()':
            i += 1
            yield ('OP', ch)
        elif ch.isspace():
            i += 1  
            pass
        elif ch.isdigit():
            res = ""
            while i < l and text[i].isdigit():
                res += text[i]
                i += 1
            yield ('NUM', int(res))
        else:
            raise ValueError(f"tokenize {text[i]}")

def apply_op(values: list[int], operators: list[str]) -> None:
    op = operators.pop()
    if op == 'u+':
        return
    elif op == 'u-':
        val = values.pop() 
        values.append(-val)
    elif op in '*/+-^':
        b = values.pop()
        a = values.pop()
        if op == '*':
            values.append(a*b)
        elif op == '/':
            values.append(a//b)
        elif op == '+':
            values.append(a+b)
        elif op == '-':
            values.append(a-b)
        elif op == '^':
            values.append(a**b)
    else:
        raise ValueError(f"apply_op {op=}")

PRECEDENCE = {
    '+': (1, 'left'),
    '-': (1, 'left'),
    '*': (2, 'left'),
    '/': (2, 'left'),
    '^': (3, 'right'),
    'u+': (4, 'right'),
    'u-': (4, 'right')
}

def parse(tokens: Generator[tuple[str, Union[str, int]]]) -> int:
    values, operators = [],[] 
    def apply() -> None:
        apply_op(values, operators)
    expect_operand = True
    for t in tokens:
        print(f"{t} {values} {operators}")
        type, value = t
        if type == 'NUM':
            values.append(value)
            expect_operand = False
        elif value == '(':
            operators.append(value)
            expect_operand = True
        elif value == ')':
            while operators[-1] != '(':
                apply()
            operators.pop()
            expect_operand = False
        elif expect_operand:
            if value in '+-':
                operators.append('u'+value)
            else:
                raise ValueError(f"parse error {value}")
        elif value in '+-*/^':
            while (operators and operators[-1] != '(' and
                (PRECEDENCE[operators[-1]][0] > PRECEDENCE[value][0] or
                (PRECEDENCE[operators[-1]][0] == PRECEDENCE[value][0] and
                 PRECEDENCE[value][1] == 'left'))):
                apply()
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