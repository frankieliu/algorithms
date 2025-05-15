# This handles unary operators in lex
# I rather this be handled by the parser
# See lalr_ds1.py

def lex(input_str):
    """Lexer that handles unary operators by distinguishing them from binary ones"""
    tokens = []
    i = 0
    n = len(input_str)
    while i < n:
        if input_str[i].isspace():
            i += 1
        elif input_str[i] in '+-*/^()':
            # Check if +/- is unary (at start or after another operator or '(')
            if input_str[i] in '+-':
                if (not tokens or 
                    tokens[-1][0] == 'OP' and tokens[-1][1] != ')' or
                    tokens[-1][1] == '('):
                    tokens.append(('UNARY', input_str[i]))
                else:
                    tokens.append(('OP', input_str[i]))
            else:
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

# Precedence rules now include unary operators
PRECEDENCE = {
    'u+': (5, 'right'),  # Unary +
    'u-': (5, 'right'),  # Unary -
    '^': (4, 'right'),
    '*': (3, 'left'),
    '/': (3, 'left'),
    '+': (2, 'left'),
    '-': (2, 'left'),
}

def parse(tokens):
    """Parser that handles unary operators with correct precedence"""
    vals = []
    ops = []
    
    def apply_op():
        op = ops.pop()
        if op in ('u+', 'u-'):  # Unary operation
            a = vals.pop()
            if op == 'u-':
                vals.append(-a)
            else:
                vals.append(a)
        else:  # Binary operation
            b = vals.pop()
            a = vals.pop()
            if op == '+': vals.append(a + b)
            elif op == '-': vals.append(a - b)
            elif op == '*': vals.append(a * b)
            elif op == '/': vals.append(a / b)
            elif op == '^': vals.append(a ** b)
    
    for token in tokens:
        type, value = token
        
        if type == 'NUM':
            vals.append(value)
        elif value == '(':
            ops.append(value)
        elif value == ')':
            while ops[-1] != '(':
                apply_op()
            ops.pop()  # Remove '('
        elif type == 'UNARY':
            # Convert to our internal unary operator representation
            op = 'u' + value
            ops.append(op)
        elif type == 'OP':
            while (ops and ops[-1] != '(' and
                   (PRECEDENCE[ops[-1]][0] > PRECEDENCE[value][0] or
                    (PRECEDENCE[ops[-1]][0] == PRECEDENCE[value][0] and
                     PRECEDENCE[value][1] == 'left'))):
                apply_op()
            ops.append(value)
    
    while ops:
        apply_op()
    
    return vals[0] if vals else None

# Test cases including unary operators
if __name__ == "__main__":
    test_cases = [
        "3 + -5 * 2",      # -7 (unary - has higher precedence)
        "-3 * 5 + 2",       # -13
        "2 ^ -3 ^ 2",       # 2^(-(3^2)) ≈ 0.00195
        "-(10 - 3) - 2",    # -9
        "5 + + + 3",        # 8 (multiple unary +)
        "5 + - - 3",        # 8 (double negation)
        "- (2 + 3) * 4",    # -20
    ]

    for expr in test_cases:
        tokens = lex(expr)
        result = parse(tokens)
        print(f"{expr} = {result}")