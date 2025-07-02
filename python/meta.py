print("hello")

# merge k sorted lists
# probability question


# Verkada
# import requests
# import mysql.connector
# import pandas as pd

# "21+3x4+1", 34  print('Hello')
prec = {
    "+": (1, "left"),
    "x": (2, "left")
}
import re

def tokenize(s: str):
    s = re.sub(r"\s+", "", s)
    i = 0
    res = []
    
    while i < len(s):
        ch = s[i]
        if ch.isnumeric():
            tmp = 0 
            while i < len(s) and s[i].isnumeric():
                tmp = tmp * 10 + int(s[i])
                i+=1
            res.append(('num', tmp))
        elif ch in "+x":
            res.append(('op', ch))
            i+=1
        else:
            raise ValueError(f"{ch} not expected")
    return res
                  
def reduce(val, op):
    b, a = val.pop(), val.pop()
    oper = op.pop()
    if oper == "+":
        a += b
    elif oper == 'x':
        a *= b
    val.append(a)
    
def parse(s: str):
    tokens = tokenize(s)
    val, op = [],[] 
    for type, value in tokens:
        if type == "num":
            val.append(value)
        elif type == "op":
            while op and prec[op[-1]][0] > prec[value][0]:
                reduce(val, op)
            op.append(value)
        print(val, op)
    while op:
        reduce(val, op)
    return val[-1]
            

# print(parse("1+2*5"))
# "3"
# "1++2"
# print(parse("21+3x4+1"))
# assert(parse("21+3x4+1") == 21+3*4+1)
# assert parse("3x4+10+5x2x3+7") == 59

def parse2(s):
    i = 0 
    val, op = [],[]
    reduce_on_next_operand = False
    while i < len(s):
        ch = s[i]
        if ch.isnumeric():
            tmp = 0 
            while i < len(s) and s[i].isnumeric():
                tmp = tmp * 10 + int(s[i])
                i+=1
            if reduce_on_next_operand:
                a = val.pop()
                opr = op.pop()
                if opr == 'x':
                    a *= tmp
                else:
                    a += tmp
                val.append(a)
                reduce_on_next_operand = False
            else:
                val.append(tmp)
        elif ch == 'x':
            reduce_on_next_operand = True
            op.append(ch)
            i+=1
        elif ch == '+':
            if len(val) == 2:
                b,a = val.pop(), val.pop()
                opr = op.pop()
                if opr == 'x':
                    a *= b
                else:
                    a += b
                val.append(a)
            op.append(ch)
            i+=1
        print(val, op)
    if len(val) == 2:
        b,a = val.pop(), val.pop() 
        if opr == 'x':
            a *= b 
        else:
            a += b
        val.append(a)
    return val[-1]

print(parse2("21+3x4+1"))
# assert(parse("21+3x4+1") == 21+3*4+1)
assert parse2("3x4+10+5x2x3+7") == 59


# 1, 0x1, 1
# 2 0x10, 1
# 3 0x11, 2
