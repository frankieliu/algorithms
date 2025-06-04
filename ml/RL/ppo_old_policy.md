<frankie>
For PPO how is the old policy \pi obtained, this is wrt to the ration term in the surrogate objective loss function
</frankie>

In **Proximal Policy Optimization (PPO)**, the **old policy (\(\pi_{\text{old}}\))** plays a critical role in the **surrogate objective function**, which uses a ratio of probabilities between the new policy (\(\pi_{\theta}\)) and the old policy to constrain updates. Here’s how \(\pi_{\text{old}}\) is obtained and used:

---

### **1. Role of the Old Policy in PPO**
The surrogate objective function in PPO is:
\[
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
\]
where:
- \( r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\text{old}}(a_t | s_t)} \) is the **probability ratio**.
- \(\pi_{\text{old}}\) is the policy **before the update** (fixed during optimization).
- \(\hat{A}_t\) is the advantage estimate.

---

### **2. How \(\pi_{\text{old}}\) is Obtained**
1. **Initialization**:
   - At the start of training, \(\pi_{\text{old}}\) is the **same as the current policy** (\(\pi_{\theta}\)) since no updates have occurred yet.

2. **During Training**:
   - **Phase 1: Data Collection**:
     - The current policy \(\pi_{\theta}\) interacts with the environment to collect trajectories (states, actions, rewards).
     - These trajectories are used to compute advantages \(\hat{A}_t\) (e.g., using GAE).
   - **Phase 2: Policy Update**:
     - \(\pi_{\text{old}}\) is **fixed** to the policy that collected the data (\(\pi_{\theta}\) before updates).
     - The surrogate objective is optimized **w.r.t. \(\pi_{\theta}\)**, but the ratio \(r_t(\theta)\) depends on the **old probabilities** \(\pi_{\text{old}}(a_t|s_t)\).
     - After each gradient step, \(\pi_{\text{old}}\) is **updated to the new \(\pi_{\theta}\)** for the next batch of data.

3. **Key Points**:
   - \(\pi_{\text{old}}\) is **not a separate network**. It’s a **snapshot** of \(\pi_{\theta}\) at the time of data collection.
   - The probabilities \(\pi_{\text{old}}(a_t|s_t)\) are **precomputed and stored** during data collection to avoid recomputation.

---

### **3. Practical Implementation**
Here’s how this works in code (PyTorch-like pseudocode):

```python
import torch

def train_ppo():
    policy = PolicyNetwork()  # Current policy π_θ
    optimizer = torch.optim.Adam(policy.parameters())
    
    for epoch in range(num_epochs):
        # Phase 1: Collect data using current policy
        states, actions, rewards, old_log_probs = collect_data(policy)
        
        # Phase 2: Update policy (π_old is fixed to the data-collecting policy)
        for _ in range(update_steps):
            # Compute new log probabilities
            new_log_probs = policy.get_log_prob(states, actions)
            
            # Ratio: exp(new_log_probs - old_log_probs)
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Surrogate loss with clipping
            advantages = compute_advantages(rewards)  # Precomputed
            surrogate_loss = -torch.min(
                ratios * advantages,
                torch.clamp(ratios, 1-clip_epsilon, 1+clip_epsilon) * advantages
            ).mean()
            
            # Update policy
            optimizer.zero_grad()
            surrogate_loss.backward()
            optimizer.step()
```

#### **Key Details**:
- `old_log_probs` are **detached from the computation graph** (no gradients flow to \(\pi_{\text{old}}\)).
- The ratio \(r_t(\theta)\) is **recomputed at each update step** using the **latest \(\pi_{\theta}\)**, but compared to the **fixed \(\pi_{\text{old}}\)**.

---

### **4. Why This Matters**
- **Stability**: By fixing \(\pi_{\text{old}}\), PPO avoids overly large policy updates (enforced by the clip term).
- **Efficiency**: No need to maintain a separate old policy network; just store log probabilities during rollout.

---

### **5. Comparison to Other Methods**
| Method       | Old Policy Handling                          | Key Difference                          |
|--------------|---------------------------------------------|----------------------------------------|
| **PPO**      | Snapshot of \(\pi_{\theta}\) at rollout     | Uses clipping to constrain updates.    |
| **TRPO**     | Computes KL divergence with \(\pi_{\text{old}}\) | More complex (second-order optimization). |
| **VPG**      | No old policy; updates directly.            | Unstable (no constraint).              |

---

### **Summary**
- \(\pi_{\text{old}}\) is a **fixed snapshot** of \(\pi_{\theta}\) taken at data collection time.
- The ratio \(\frac{\pi_\theta}{\pi_{\text{old}}}\) ensures updates are **constrained** (via clipping).
- Implementationally, you only need to **store log probabilities** from the rollout phase.

Would you like a full PPO implementation for a ranking task?