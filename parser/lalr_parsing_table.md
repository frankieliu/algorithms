https://www.geeksforgeeks.org/lalr-parser-with-examples/

May 26th 2025, 8:17:47 am

### You

how does one construct a LR parsing table

### ChatGPT

Constructing an **LR parsing table** is a structured process used in compiler design to implement **bottom-up parsers**, such as LR(0), SLR(1), LR(1), or LALR(1) parsers. Here’s a step-by-step guide focusing on how to construct an **LR(0)** parsing table. More advanced versions (SLR, LR(1), LALR) follow similar steps with more context (lookaheads).

* * *

## 🔹 Step-by-Step: Constructing an LR(0) Parsing Table

### **1\. Augment the Grammar**

Add a new start symbol $S'→S$, where $S$ is the original start symbol.

### **2\. Construct the Canonical Collection of LR(0) Items**

Each item is a production with a dot `•` indicating how much of the production has been seen:

-   Example: $A \to \cdot \alpha$ means none of $\alpha$ has been parsed yet.
    
-   $A \to \alpha \cdot$ means the entire right-hand side has been parsed.
    

#### Steps:

-   Start with the **closure** of the initial item $S' \to \cdot S$.
    
-   Repeatedly apply **closure** and **goto** to generate all the LR(0) states:
    
    -   `closure(I)`: For each item $A \to \alpha \cdot B\beta$, add all productions $B \to \cdot \gamma$
        
    -   `goto(I, X)`: Move the dot over $X$ in all items and compute the closure
        

You’ll end up with a **DFA** of LR(0) item sets (states).

### **3\. Construct the Parsing Table**

You construct two tables:

-   **ACTION** table: For terminals (shift, reduce, accept, error)
    
-   **GOTO** table: For nonterminals (state transitions)
    

#### For each state (item set III):

-   If A→α⋅aβ∈IA \\to \\alpha \\cdot a\\beta \\in IA→α⋅aβ∈I and goto(I,a)\=J\\text{goto}(I, a) = Jgoto(I,a)\=J, and aaa is a terminal:
    
    -   Set `ACTION[I, a] = shift J`
        
-   If A→α⋅∈IA \\to \\alpha \\cdot \\in IA→α⋅∈I, i.e., a complete item:
    
    -   If A≠S′A \\ne S'A\=S′, set `ACTION[I, a] = reduce A → α` for **all terminals** aaa
        
    -   If A\=S′→S⋅A = S' \\to S \\cdotA\=S′→S⋅, set `ACTION[I, $] = accept`
        
-   For a non-terminal AAA, if goto(I,A)\=J\\text{goto}(I, A) = Jgoto(I,A)\=J, set `GOTO[I, A] = J`
    

> In **SLR(1)**, you only allow the reduce action on FOLLOW(A) instead of all terminals.  
> In **LR(1)**, items are of the form A→α⋅β,aA \\to \\alpha \\cdot \\beta, aA→α⋅β,a, and lookaheads are tracked for more precision.

* * *

## 🔹 Example: Grammar

text

CopyEdit

```
S' → S
S  → CC
C  → cC | d
```

### Step 1: Augmented Grammar

text

CopyEdit

```
0: S' → •S
1: S  → •CC
2: C  → •cC
3: C  → •d
```

### Step 2: Canonical Collection of LR(0) Items

Construct states I0,I1,...,InI\_0, I\_1, ..., I\_nI0​,I1​,...,In​ using closure and goto (manually or with automation).

### Step 3: Parsing Table

Fill in ACTION and GOTO from the DFA.

* * *

## 🔹 Tools

For larger grammars, manually constructing LR tables is tedious. Tools like:

-   Bison
    
-   [ANTLR](https://www.antlr.org/)
    
-   Educational platforms (e.g., JFLAP)
    

can generate parsing tables and help visualize the DFA.

* * *

Would you like to walk through a full example with states and the full parsing table?

### You

Can you walk through a full example with states and the full parsing table

### ChatGPT

Absolutely! Let's walk through a **complete LR(0) example** with:

* * *

### 📘 Grammar

We'll use a simple grammar:

vbnet

CopyEdit

```
S' → S         (Augmented)
S  → CC
C  → cC | d
```

* * *

## 🔹 Step 1: Augment the Grammar

Add a new start production:

makefile

CopyEdit

```
0: S' → •S
1: S  → •CC
2: C  → •cC
3: C  → •d
```

* * *

## 🔹 Step 2: Construct the Canonical Collection of LR(0) Items

We’ll construct the **DFA of LR(0) item sets**: closures and transitions.

* * *

### 🔸 I₀: closure(S' → •S)

mathematica

CopyEdit

```
S' → •S
S  → •CC
C  → •cC
C  → •d
```

Transitions from I₀:

-   On **S** → I₁
    
-   On **C** → I₂
    
-   On **c** → I₃
    
-   On **d** → I₄
    

* * *

### 🔸 I₁: closure(S' → S•)

vbnet

CopyEdit

```
S' → S•
```

✅ Accepting state if input is done

* * *

### 🔸 I₂: closure(S → C•C)

mathematica

CopyEdit

```
S → C•C
C → •cC
C → •d
```

From I₂:

-   On **C** → I₅
    
-   On **c** → I₃
    
-   On **d** → I₄
    

* * *

### 🔸 I₃: closure(C → c•C)

mathematica

CopyEdit

```
C → c•C
C → •cC
C → •d
```

From I₃:

-   On **C** → I₆
    
-   On **c** → I₃
    
-   On **d** → I₄
    

* * *

### 🔸 I₄: closure(C → d•)

mathematica

CopyEdit

```
C → d•
```

✅ Reduction item: reduce by `C → d`

* * *

### 🔸 I₅: closure(S → CC•)

nginx

CopyEdit

```
S → CC•
```

✅ Reduction item: reduce by `S → CC`

* * *

### 🔸 I₆: closure(C → cC•)

mathematica

CopyEdit

```
C → cC•
```

✅ Reduction item: reduce by `C → cC`

* * *

## 🔹 Step 3: Build the LR(0) Parsing Table

We’ll define:

-   ACTION: for terminals `c`, `d`, and `$` (end of input)
    
-   GOTO: for nonterminals `S`, `C`
    

Let’s number the states:

| State | Items |
| --- | --- |
| I₀ | S' → •S, S → •CC, C → •cC, C → •d |
| I₁ | S' → S• |
| I₂ | S → C•C, C → •cC, C → •d |
| I₃ | C → c•C, C → •cC, C → •d |
| I₄ | C → d• |
| I₅ | S → CC• |
| I₆ | C → cC• |

* * *

### Transitions Summary

| From | Symbol | To |
| --- | --- | --- |
| I₀ | S | I₁ |
| I₀ | C | I₂ |
| I₀ | c | I₃ |
| I₀ | d | I₄ |
| I₂ | C | I₅ |
| I₃ | C | I₆ |
| I₃ | c | I₃ |
| I₃ | d | I₄ |

* * *

### Parsing Table

Let’s build the ACTION and GOTO tables:

#### ACTION table

| State | c | d | $ |
| --- | --- | --- | --- |
| 0 | s3 | s4 |  |
| 1 |  |  | accept |
| 2 | s3 | s4 |  |
| 3 | s3 | s4 |  |
| 4 | r3 (C→d) | r3 | r3 |
| 5 | r1 (S→CC) | r1 | r1 |
| 6 | r2 (C→cC) | r2 | r2 |

#### GOTO table

| State | S | C |
| --- | --- | --- |
| 0 | 1 | 2 |
| 1 |  |  |
| 2 |  | 5 |
| 3 |  | 6 |
| 4 |  |  |
| 5 |  |  |
| 6 |  |  |

* * *

## ✅ Final Result: LR(0) Parsing Table

| State | ACTION |  |  | GOTO |  |
| --- | --- | --- | --- | --- | --- |
|  | c | d | $ | S | C |
| 0 | s3 | s4 |  | 1 | 2 |
| 1 |  |  | accept |  |  |
| 2 | s3 | s4 |  |  | 5 |
| 3 | s3 | s4 |  |  | 6 |
| 4 | r3: C → d | r3 | r3 |  |  |
| 5 | r1: S → CC | r1 | r1 |  |  |
| 6 | r2: C → cC | r2 | r2 |  |  |

* * *

Let me know if you want to simulate parsing an input string (e.g. `c d d`) using this table!

### You

Can you simulate parsing an input string using this table

### ChatGPT

Absolutely! Let's simulate the **LR(0) parsing process** for the input string:

css

CopyEdit

```
Input: c d d $
```

Where `$` is the **end-of-input marker**.

* * *

## 🔹 Recap of Grammar

makefile

CopyEdit

```
0: S' → S       (Augmented)
1: S  → CC
2: C  → cC
3: C  → d
```

* * *

## 🔹 LR(0) Parsing Table

| State | ACTION |  |  | GOTO |  |
| --- | --- | --- | --- | --- | --- |
|  | c | d | $ | S | C |
| 0 | s3 | s4 |  | 1 | 2 |
| 1 |  |  | accept |  |  |
| 2 | s3 | s4 |  |  | 5 |
| 3 | s3 | s4 |  |  | 6 |
| 4 | r3: C → d | r3 | r3 |  |  |
| 5 | r1: S → CC | r1 | r1 |  |  |
| 6 | r2: C → cC | r2 | r2 |  |  |

* * *

## 🔹 Parsing Simulation

We’ll track 4 columns:

| Step | Stack | Input | Action | Notes |
| --- | --- | --- | --- | --- |
| 0 | `[0]` | `c d d $` | `ACTION[0, c]=s3` | shift c, go to state 3 |
| 1 | `[0, c, 3]` | `d d $` | `ACTION[3, d]=s4` | shift d, go to state 4 |
| 2 | `[0, c, 3, d, 4]` | `d $` | `r3: C → d` | reduce by C → d |
| 3 | `[0, c, 3, C]` | `d $` | `GOTO[3, C]=6` | GOTO from state 3 on C |
| 4 | `[0, c, 3, C, 6]` | `d $` | `r2: C → cC` | reduce by C → cC |
| 5 | `[0, C]` | `d $` | `GOTO[0, C]=2` |  |
| 6 | `[0, C, 2]` | `d $` | `ACTION[2, d]=s4` | shift d, go to state 4 |
| 7 | `[0, C, 2, d, 4]` | `$` | `r3: C → d` | reduce by C → d |
| 8 | `[0, C, 2, C]` | `$` | `GOTO[2, C]=5` |  |
| 9 | `[0, C, 2, C, 5]` | `$` | `r1: S → CC` | reduce by S → CC |
| 10 | `[0, S]` | `$` | `GOTO[0, S]=1` |  |
| 11 | `[0, S, 1]` | `$` | `accept` | ✅ input accepted |

* * *

## ✅ Final Result

The input `c d d` is successfully **parsed** by this LR(0) parser. 🎉

Would you like to see a visualization of this parsing stack as a table or diagram?