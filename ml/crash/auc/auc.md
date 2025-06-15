### **Understanding AUC Calculation from Scratch**

The **Area Under the ROC Curve (AUC)** measures how well a binary classifier ranks positive instances higher than negative ones. A perfect classifier (AUC = 1.0) ranks all positives above all negatives, while a random classifier (AUC = 0.5) has no ranking ability.

---

## **Key Intuition Behind AUC**
AUC is equivalent to the probability that a randomly chosen **positive** instance is ranked higher than a randomly chosen **negative** instance.  

Mathematically:  
\[
\text{AUC} = P(\text{score of positive} > \text{score of negative})
\]

If two scores are **tied**, we count them as **half-correct** (i.e., probability = 0.5).

---

## **Step-by-Step Calculation**
### **1. Sort Predictions by Score (Descending)**
We sort the predictions in **decreasing order** of predicted scores.  

**Example Data:**
| Index | True Label (y_true) | Predicted Score (y_score) |
|-------|---------------------|---------------------------|
| 0     | 0                   | 0.1                       |
| 1     | 0                   | 0.3                       |
| 2     | 1                   | 0.35                      |
| 3     | 1                   | 0.4                       |
| 4     | 0                   | 0.4                       | *(tied with index 3)* |
| 5     | 1                   | 0.5                       |
| 6     | 0                   | 0.5                       | *(tied with index 5)* |
| 7     | 1                   | 0.7                       |
| 8     | 1                   | 0.8                       |
| 9     | 1                   | 0.9                       |

**After Sorting:**
| Index | True Label | Score |
|-------|------------|-------|
| 9     | 1          | 0.9   |
| 8     | 1          | 0.8   |
| 7     | 1          | 0.7   |
| 5     | 1          | 0.5   |
| 6     | 0          | 0.5   | *(tied with 5)* |
| 3     | 1          | 0.4   |
| 4     | 0          | 0.4   | *(tied with 3)* |
| 2     | 1          | 0.35  |
| 1     | 0          | 0.3   |
| 0     | 0          | 0.1   |

---

### **2. Assign Ranks (Handling Ties)**
- If multiple predictions have the **same score**, they get the **average rank** of their positions.
- Ranks start at **1** (highest score).

**Example:**
- Scores `[0.5, 0.5]` are at positions **4 & 5** → Average rank = `(4 + 5) / 2 = 4.5`
- Scores `[0.4, 0.4]` are at positions **6 & 7** → Average rank = `(6 + 7) / 2 = 6.5`

| Index | True Label | Score | Rank |
|-------|------------|-------|------|
| 9     | 1          | 0.9   | 1    |
| 8     | 1          | 0.8   | 2    |
| 7     | 1          | 0.7   | 3    |
| 5     | 1          | 0.5   | 4.5  | *(tied with 6)* |
| 6     | 0          | 0.5   | 4.5  | *(tied with 5)* |
| 3     | 1          | 0.4   | 6.5  | *(tied with 4)* |
| 4     | 0          | 0.4   | 6.5  | *(tied with 3)* |
| 2     | 1          | 0.35  | 8    |
| 1     | 0          | 0.3   | 9    |
| 0     | 0          | 0.1   | 10   |

---

### **3. Sum Ranks of Positive Instances**
\[
\text{Rank Sum} = \sum (\text{Ranks where } y_{\text{true}} = 1)
\]

From the table:
- Positive instances (label=1) have ranks: `1, 2, 3, 4.5, 6.5, 8`  
- **Rank Sum** = `1 + 2 + 3 + 4.5 + 6.5 + 8 = 25`

---

### **4. Compute AUC Using Mann-Whitney U Statistic**
\[
\text{AUC} = \frac{\text{Rank Sum} - \frac{n_{\text{pos}} (n_{\text{pos}} + 1)}{2}}{n_{\text{pos}} \times n_{\text{neg}}}
\]

Where:
- \( n_{\text{pos}} \) = Number of positive samples (`6` in our example)
- \( n_{\text{neg}} \) = Number of negative samples (`4` in our example)

**Calculation:**
\[
\text{AUC} = \frac{25 - \frac{6 \times (6 + 1)}{2}}{6 \times 4} = \frac{25 - 21}{24} = \frac{4}{24} = 0.1667
\]

**Wait, this seems wrong!**  
AUC should be between `0.5` (random) and `1.0` (perfect). What’s the issue?

---

### **5. Fixing the Interpretation**
The problem is that **higher ranks** (lower numbers) correspond to **higher scores**. To fix this:
- **Reverse the ranking** (assign rank `1` to the **lowest score**).
- Or, **use the formula correctly** (as implemented in the code).

The correct formula (as in the code) is:
\[
\text{AUC} = \frac{\text{Rank Sum} - \frac{n_{\text{pos}} (n_{\text{pos}} + 1)}{2}}{n_{\text{pos}} \times n_{\text{neg}}}
\]
where:
- **Rank Sum** = Sum of ranks of **positive samples** (with rank `1` = lowest score).

**Recomputing ranks (ascending order):**
| Index | True Label | Score | Rank |
|-------|------------|-------|------|
| 0     | 0          | 0.1   | 1    |
| 1     | 0          | 0.3   | 2    |
| 2     | 1          | 0.35  | 3    |
| 3     | 1          | 0.4   | 4.5  | *(tied with 4)* |
| 4     | 0          | 0.4   | 4.5  | *(tied with 3)* |
| 5     | 1          | 0.5   | 6.5  | *(tied with 6)* |
| 6     | 0          | 0.5   | 6.5  | *(tied with 5)* |
| 7     | 1          | 0.7   | 8    |
| 8     | 1          | 0.8   | 9    |
| 9     | 1          | 0.9   | 10   |

**Rank Sum (positives)** = `3 + 4.5 + 6.5 + 8 + 9 + 10 = 41`  
**AUC** = \( \frac{41 - \frac{6 \times 7}{2}}{6 \times 4} = \frac{41 - 21}{24} = \frac{20}{24} = 0.8333 \)

**This matches `sklearn.metrics.roc_auc_score`!**

---

## **Final Formula**
\[
\boxed{\text{AUC} = \frac{\text{(Sum of positive ranks)} - \frac{n_{\text{pos}} (n_{\text{pos}} + 1)}{2}}{n_{\text{pos}} \times n_{\text{neg}}}}
\]

### **Key Takeaways**
1. **AUC measures ranking ability**, not just classification accuracy.
2. **Ties are handled by assigning average ranks**.
3. The formula is derived from the **Mann-Whitney U test**.
4. **Higher AUC (closer to 1.0)** means better ranking of positives over negatives.

This matches how `sklearn` computes AUC, ensuring correctness! 🚀