def lex(input_str):
    """Simpler lexer that doesn't distinguish unary operators"""
    tokens = []
    i = 0
    n = len(input_str)
    while i < n:
        if input_str[i].isspace():
            i += 1
        elif input_str[i] in '+-*/^()':
            tokens.append(('OP', input_str[i]))
            i += 1
        elif input_str[i].isdigit():
            num = ''
            while i < n and input_str[i].isdigit():
                num += input_str[i]
                i += 1
            tokens.append(('NUM', int(num)))
        else:
            raise ValueError(f"Unknown character: {input_str[i]}")
    return tokens

# Precedence rules including unary operators
PRECEDENCE = {
    'u-': (5, 'right'),  # Unary negation (highest precedence)
    'u+': (5, 'right'),  # Unary plus
    '^': (4, 'right'),
    '*': (3, 'left'),
    '/': (3, 'left'),
    '+': (2, 'left'),
    '-': (2, 'left'),
}

def parse(tokens):
    """Parser that determines unary/binary operator status"""
    values = []
    operators = []
    
    def apply_operator():
        op = operators.pop()
        if op in ('u+', 'u-'):  # Unary operator
            a = values.pop()
            values.append(a if op == 'u+' else -a)
        else:  # Binary operator
            b = values.pop()
            a = values.pop()
            if op == '+': values.append(a + b)
            elif op == '-': values.append(a - b)
            elif op == '*': values.append(a * b)
            elif op == '/': values.append(a / b)
            elif op == '^': values.append(a ** b)
    
    # Track parser state to determine if next operator is unary
    expect_operand = True
    
    for token in tokens:
        type, value = token
        
        if type == 'NUM':
            values.append(value)
            expect_operand = False
        elif value == '(':
            operators.append(value)
            expect_operand = True
        elif value == ')':
            while operators[-1] != '(':
                apply_operator()
            operators.pop()  # Remove '('
            expect_operand = False
        elif type == 'OP':
            # Determine if this is a unary operator
            if value in '+-' and expect_operand:
                op = 'u' + value  # Mark as unary
            else:
                op = value
            
            # Apply higher precedence operators first
            while (operators and operators[-1] != '(' and
                   (PRECEDENCE[operators[-1]][0] > PRECEDENCE[op][0] or
                    (PRECEDENCE[operators[-1]][0] == PRECEDENCE[op][0] and
                     PRECEDENCE[op][1] == 'left'))):
                apply_operator()
            
            operators.append(op)
            if op in ('u+', 'u-'):
                expect_operand = True  # Unary ops expect an operand after
            else:
                expect_operand = True  # Binary ops expect an operand after
    
    while operators:
        apply_operator()
    
    return values[0] if values else None

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
        tokens = lex(expr)
        result = parse(tokens)
        print(f"{expr:15} = {result}")