<hr style="border-color: blue; border-width: 5px; border-style: solid;">

<frankie>
can you explain the formula for U1, why does it give the number of pair rankings where positive are bigger than negative
</frankie>

---
### **Understanding the U₁ Formula: Why It Counts Positive-Negative Dominance**

The **Mann-Whitney U₁ statistic** is defined as:
\[
U_1 = R_1 - \frac{n_1(n_1 + 1)}{2}
\]
where:
- \( R_1 \) = Sum of ranks of **positive samples** (when all samples are sorted),
- \( n_1 \) = Number of positive samples,
- \( \frac{n_1(n_1 + 1)}{2} \) = Minimum possible rank sum for positives.

**Key Question**:  
*Why does this formula compute the number of times positives outrank negatives?*

---

## **1. Intuition: U₁ Counts "Wins" for Positives**
Imagine two groups:  
- **Positives (n₁ samples)**  
- **Negatives (n₂ samples)**  

We want to count how many **(positive, negative) pairs** have:  
\[
\text{score}_{\text{positive}} > \text{score}_{\text{negative}}
\]

This is equivalent to counting **dominant pairs** where positives "beat" negatives.

---

## **2. Step-by-Step Derivation**
### **Step 1: Rank All Samples**
- Sort **all \( n_1 + n_2 \)** samples by their scores (ascending: rank 1 = lowest score).
- Assign **average ranks for ties**.

**Example**:  
| Sample | Label (y) | Score | Rank |
|--------|----------|-------|------|
| A      | 0        | 0.1   | 1    |
| B      | 0        | 0.3   | 2    |
| C      | 1        | 0.4   | 3    |
| D      | 1        | 0.5   | 4    |
| E      | 1        | 0.8   | 5    |

Here:
- \( n_1 = 3 \) (positives: C, D, E),  
- \( n_2 = 2 \) (negatives: A, B).

### **Step 2: Sum Ranks of Positives (\( R_1 \))**
\[
R_1 = \text{Rank}(C) + \text{Rank}(D) + \text{Rank}(E) = 3 + 4 + 5 = 12
\]

### **Step 3: Compute Minimum Possible \( R_1 \)**
If **all positives were ranked lowest** (worst-case scenario), their ranks would be \( 1, 2, \dots, n_1 \):  
\[
\text{Min}(R_1) = 1 + 2 + 3 = 6 \quad \text{(for } n_1 = 3)
\]

### **Step 4: \( U_1 \) Measures Excess Ranking**
\[
U_1 = R_1 - \text{Min}(R_1) = 12 - 6 = 6
\]

### **Step 5: Verify \( U_1 \) Counts Dominant Pairs**
- **Total possible pairs**: \( n_1 n_2 = 3 \times 2 = 6 \).  
- **Actual dominant pairs**:  
  - (C > A), (C > B)  
  - (D > A), (D > B)  
  - (E > A), (E > B)  
  → **6 pairs** where positives > negatives.  

Thus, \( U_1 = 6 \) correctly counts these pairs!

---

## **3. Why Does This Work?**
### **Key Insight**:
- The term \( R_1 \) **embeds information about how positives compare to negatives** via their ranks.  
- Subtracting \( \text{Min}(R_1) \) removes the **baseline rank sum** (if positives were all at the bottom).  
- The difference \( (R_1 - \text{Min}(R_1)) \) directly counts **how many negatives each positive "jumps over"** in the ranking.

### **Mathematical Proof**:
1. Each positive sample at rank \( r \) "dominates" **all negatives ranked below it**.  
   - If a positive is at rank \( r \), there are \( r - \text{(number of positives above it)} \) negatives below it.  
2. Summing over all positives:  
   \[
   U_1 = \sum_{\text{positives}} (\text{rank} - \text{position among positives})
   \]
   This simplifies to \( R_1 - \text{Min}(R_1) \).

---

## **4. Edge Case: Tied Scores**
If a positive and negative have the **same score**, they receive the **same average rank**.  
- This reduces \( U_1 \) by **0.5 per tied pair**, reflecting uncertainty.  

**Example**:  
If **C (positive)** and **B (negative)** both had score **0.3**:  
- Their ranks become \( \frac{2 + 3}{2} = 2.5 \).  
- Now \( R_1 = 2.5 + 4 + 5 = 11.5 \),  
- \( U_1 = 11.5 - 6 = 5.5 \).  

This matches intuition:  
- **5 strict wins** (D/E > A/B) + **0.5 for the tie** (C vs. B) = **5.5**.

---

## **5. Final Interpretation**
\[
U_1 = R_1 - \frac{n_1(n_1 + 1)}{2}
\]
- **\( R_1 \)**: Observed ranking advantage of positives.  
- **\( \frac{n_1(n_1 + 1)}{2} \)**: Baseline if positives were all at the bottom.  
- **\( U_1 \)**: Counts how many negatives are "outranked" by positives.  

Thus, \( U_1 \) **directly computes** the number of positive-negative pairs where positives win, with **ties counted as 0.5**.

---

## **Summary**
- The formula \( U_1 = R_1 - \frac{n_1(n_1 + 1)}{2} \) works because:  
  1. **Ranks encode pairwise comparisons implicitly**.  
  2. Subtracting the **minimum possible \( R_1 \)** isolates the "excess" due to positives outranking negatives.  
  3. **Ties are handled** via average ranks, ensuring fairness.  

This is why AUC (derived from \( U_1 \)) is a robust metric for ranking performance! 🚀