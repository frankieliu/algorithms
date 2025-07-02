from unittest import TestCase
from lalr_ds3 import parse

class ParserTest(TestCase):
    def test_parser(self):
        test_cases = [
            ("3 + -5 * 2",       -7),
            ("-3 * 5 + 2",       -13),
            ("2 ^ -3 ^ 2",       2**((-3)**2)), # ≈ 0.00195
            ("-(10 - 3) - 2",    -9),
            ("5 + + + 3",        8),
            ("5 + - - 3",        8),
            ("- (2 + 3) * 4",    -20),
            ("3*-2",             -6),
            ("3+-2",             1),
        ]

        for expr, expected in test_cases:
            result = parse(expr)
            self.assertEqual(result, expected)