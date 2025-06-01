### **How Maximizing Log Probability is Equivalent to Minimizing KL Divergence**

At first glance, **maximizing log probability** (Maximum Likelihood Estimation, or MLE) and **minimizing Kullback-Leibler (KL) Divergence** seem like different objectives. However, they are deeply connected—in fact, **MLE is a special case of minimizing KL divergence** when the true data distribution is represented empirically. Here’s how:

---

## **1. Definitions**
### **(A) Maximum Likelihood Estimation (MLE)**
Given a model \( P_\theta(x) \) with parameters \( \theta \) and observed data \( \{x_1, ..., x_N\} \), MLE seeks:
\[
\theta^* = \arg\max_\theta \sum_{i=1}^N \log P_\theta(x_i).
\]

### **(B) KL Divergence**
The KL divergence measures how much one probability distribution \( Q \) diverges from another \( P \):
\[
D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}.
\]
- \( D_{KL} \geq 0 \), with \( D_{KL} = 0 \) iff \( P = Q \).

---

## **2. Connecting MLE to KL Divergence**
### **Step 1: Empirical Data Distribution**
Assume the true data distribution \( P_{\text{data}}(x) \) is approximated by the empirical distribution (i.e., the dataset):
\[
P_{\text{data}}(x) = \frac{1}{N} \sum_{i=1}^N \delta(x - x_i),
\]
where \( \delta \) is the Dirac delta function (1 if \( x = x_i \), else 0).

### **Step 2: Minimizing KL Divergence Between \( P_{\text{data}} \) and \( P_\theta \)**
We want \( P_\theta \) to match \( P_{\text{data}} \), so we minimize:
\[
D_{KL}(P_{\text{data}} \| P_\theta) = \sum_x P_{\text{data}}(x) \log \frac{P_{\text{data}}(x)}{P_\theta(x)}.
\]

Substitute \( P_{\text{data}} \):
\[
D_{KL} = \sum_x \left( \frac{1}{N} \sum_{i=1}^N \delta(x - x_i) \right) \log \frac{\frac{1}{N} \sum_{j=1}^N \delta(x - x_j)}{P_\theta(x)}.
\]

Simplify (only \( x = x_i \) terms survive due to \( \delta \)):
\[
D_{KL} = \frac{1}{N} \sum_{i=1}^N \log \frac{1/N}{P_\theta(x_i)} = \frac{1}{N} \sum_{i=1}^N \left[ \log 1 - \log N - \log P_\theta(x_i) \right].
\]

Since \( \log 1 = 0 \) and \( \log N \) is constant w.r.t. \( \theta \):
\[
D_{KL} = -\frac{1}{N} \sum_{i=1}^N \log P_\theta(x_i) + \text{constant}.
\]

### **Step 3: Equivalence to MLE**
Minimizing \( D_{KL} \) is equivalent to minimizing:
\[
-\frac{1}{N} \sum_{i=1}^N \log P_\theta(x_i),
\]
which is the same as **maximizing the log likelihood**:
\[
\arg\min_\theta D_{KL}(P_{\text{data}} \| P_\theta) = \arg\max_\theta \sum_{i=1}^N \log P_\theta(x_i).
\]

---

## **3. Intuition**
- **MLE** tries to make \( P_\theta(x_i) \) large for observed data \( x_i \).
- **KL minimization** tries to make \( P_\theta \) match the empirical distribution \( P_{\text{data}} \).
- Since \( P_{\text{data}} \) is concentrated at the observed data points, the two objectives align.

---

## **4. Key Implications**
1. **MLE is KL Minimization with Empirical Data**  
   MLE assumes the dataset perfectly represents \( P_{\text{data}} \). If the dataset is small or biased, MLE may overfit.

2. **Beyond MLE: Regularization**  
   Pure MLE can lead to overconfidence. Alternatives like:
   - **Maximum A Posteriori (MAP)**: Adds a prior (equivalent to KL + regularization).
   - **Variational Inference**: Minimizes KL between approximate and true posterior.

3. **Cross-Entropy = KL + Entropy**  
   Cross-entropy \( H(P_{\text{data}}, P_\theta) \) relates to KL:
   \[
   H(P_{\text{data}}, P_\theta) = D_{KL}(P_{\text{data}} \| P_\theta) + H(P_{\text{data}}),
   \]
   where \( H(P_{\text{data}}) \) is constant. Thus, minimizing cross-entropy ≈ minimizing KL.

---

## **5. Example: Categorical Distribution**
Suppose:
- True \( P_{\text{data}} \): 60% "cat", 40% "dog".
- Model \( P_\theta \): Predicts \( \theta \) for "cat", \( 1-\theta \) for "dog".

**MLE**: Maximize \( \log P_\theta(\text{data}) = \sum_i \log P_\theta(x_i) \).  
**KL Minimization**: Minimize \( D_{KL}(P_{\text{data}} \| P_\theta) \).  

Both yield the same solution: \( \theta = 0.6 \).

---

## **6. When They Differ**
If:
- The model family \( P_\theta \) cannot perfectly fit \( P_{\text{data}} \) (e.g., underparameterized).
- We use a different divergence (e.g., reverse KL \( D_{KL}(P_\theta \| P_{\text{data}}) \), as in variational inference).

---

## **Summary**
- **Maximizing log likelihood** = **Minimizing KL divergence** between empirical data distribution and model.
- This equivalence is why:
  - Neural networks use **cross-entropy loss** (derived from KL).
  - MLE is the foundation of many ML algorithms.

Would you like a PyTorch example showing this equivalence in practice?