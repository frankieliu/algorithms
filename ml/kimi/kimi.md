# from Gemini
The paper "KIMI K1.5: Scaling Reinforcement Learning with LLMs" introduces Kimi k1.5, a new multi-modal Large Language Model (LLM) trained using **Reinforcement Learning (RL)**. The Kimi team aims to overcome the data limitations of traditional LLM pretraining by allowing the model to generate its own training data through exploration with rewards.

---

### Key Contributions and Techniques

The paper highlights several core ingredients in the development of Kimi k1.5:

* **Long Context Scaling:** Kimi k1.5 scales the context window for RL training to 128k tokens. The research indicates that increased context length leads to continuous performance improvements. A key innovation here is the use of **partial rollouts** for improved training efficiency, which involves reusing large portions of previous trajectories to sample new ones, reducing the cost of re-generating from scratch.
* **Improved Policy Optimization:** The model uses a variant of **online mirror descent** for robust policy optimization, specifically formulated for long-Chain of Thought (CoT) scenarios. This is further enhanced by effective sampling strategies, the introduction of a **length penalty**, and optimized data recipes.

* **Simplistic Framework:** Unlike prior approaches that often rely on complex techniques like Monte Carlo tree search, value functions, or process reward models, Kimi k1.5 achieves strong performance with a more straightforward RL framework. The ability to scale context length allows the learned CoTs to exhibit properties of planning, reflection, and correction.
* **Multimodality:** Kimi k1.5 is trained on both text and vision data, enabling it to reason jointly across these modalities.
* **Long2short Methods:** The paper also presents "long2short" methods, which leverage techniques developed for long-CoT models to enhance the performance of short-CoT models. These methods include **model merging**, **shortest rejection sampling**, **Direct Preference Optimization (DPO)**, and a dedicated **long2short RL** phase.

---

### RL Training Details

The RL training process for Kimi k1.5 involves:

* **RL Prompt Set Curation:** The team emphasizes the importance of a high-quality, diverse, balanced, and accurately evaluable prompt set to prevent reward hacking and overfitting. They use automatic filters, a model-based difficulty assessment, and methods to identify and remove "easy-to-hack" prompts.
* **Long-CoT Supervised Fine-Tuning (SFT):** A small, high-quality long-CoT warm-up dataset is created through prompt engineering, focusing on human-like reasoning processes such as planning, evaluation, reflection, and exploration. This SFT primes the model for generating detailed and logically coherent responses.
* **Policy Optimization (Detailed):** The core RL objective is to maximize the reward for correct answers. The training algorithm is a variant of online policy mirror descent. Notably, the system *excludes a value network* from its training. The authors hypothesize that traditional value functions might not be suitable for encouraging exploration of diverse reasoning paths, which is crucial for developing robust problem-solving strategies.
* **Length Penalty:** To counteract "overthinking" (excessive response length) during RL training, a length reward is introduced. This reward promotes shorter, correct responses and penalizes longer, incorrect ones, with a warm-up phase to avoid hindering initial training.
* **Sampling Strategies:** To improve training efficiency, Kimi k1.5 employs **curriculum sampling** (starting with easier tasks and progressing to harder ones) and **prioritized sampling** (focusing on problems where the model underperforms based on its success rate).
* **Test Case Generation for Coding:** For coding problems without available test cases, the paper describes an automated method to generate high-quality test cases using the CYaRon library and leveraging ground truth solutions.
* **Reward Modeling for Math:** To accurately evaluate mathematical solutions, the paper discusses two approaches: a classic value-head based reward model and a more accurate **Chain-of-Thought (CoT) Reward Model**, which explicitly generates a step-by-step reasoning process before providing a correctness judgment. The CoT Reward Model achieved significantly higher accuracy.
* **Vision Data:** The vision RL data is sourced from real-world data (science, location guessing, data analysis), synthetic visual reasoning data (spatial relationships, patterns), and text-rendered data (converting text to images to ensure consistency).

---

### Performance

Kimi k1.5 demonstrates **state-of-the-art reasoning performance** across multiple benchmarks and modalities:

* **Long-CoT results:** It achieves scores like 77.5 on AIME, 96.2 on MATH 500, 94th percentile on Codeforces, and 74.9 on MathVista, matching or exceeding models like OpenAI's o1.
* **Short-CoT results:** Through its long2short methods, Kimi k1.5 also achieves state-of-the-art short-CoT reasoning performance, including 60.8 on AIME, 94.6 on MATH500, and 47.3 on LiveCodeBench, significantly outperforming existing short-CoT models such as GPT-4o and Claude Sonnet 3.5 (up to +550%).

---

### Training Stages

The overall development of Kimi k1.5 consists of several stages:

1.  **Pretraining:** A multi-stage process involving vision-language pretraining, cooldown for consolidating capabilities, and long-context activation up to 131,072 tokens.
2.  **Vanilla Supervised Fine-tuning (SFT):** Building a corpus for non-reasoning tasks.
3.  **Long-CoT Supervised Fine-tuning.**
4.  **Reinforcement Learning.**

---

The paper concludes that Kimi k1.5 establishes a simplistic yet effective RL framework by leveraging long context scaling and improved policy optimization, demonstrating that LLMs can effectively scale their training data through learning to explore with rewards, leading to significant advancements in reasoning capabilities.

---

Kimi:

# Long context scaling
- 128k token context window
- expensive to do a single rollout
- partial rollout reuse significant portion from previously generated sequence

# Mirror Descent  
- optimizing over probablity distributions
  - values are non-negative and sum up to one
- instead, perform updates in dual space and map it back to primal space
- transformation via a mirror map
  - aka distance generating function
  - aka potential function

## Advantages
- faster convergence
- implicit regularization
- avoid value functions

## Steps
1. map parameters to dual space
2. gradient step in dual space
3. map back to primal space
4. projection if constrained

## Example mapping
1. $\phi(x)  = x \log x$
1. $\grad \phi = \log x + 1$
1. Can think of $\log$ as mapping probabilities to log probability

## LLM
1. policy \pi(y,z|x) represent the probability of generating a z (CoT) and final answer y, given input x.


# Problem statement
- Need more data to scale models "intelligence"

# Features
- 128k token context length
- online mirror gradient descent
- sampling strategy, length penalty
- simple framework
  - no Monte Carlo tree search
  - value functions
  - process rewards
- multimodality

# Training data:

$\mathcal{D}=\left\{\left(x_{i}, y_{i}^{*}\right)\right\}_{i=1}^{n}$

problems: $x_{i}$ 
answers: $y_{i}^{*}$

# Goal:

Find policy $\pi_{\theta}$

# CoT to the rescue:
- use a sequence of intermediate steps $z=\left(z_{1}, z_{2}, \ldots, z_{m}\right)$
- bridges $x$ and $y$
- $y, z \sim \pi_{\theta}$

# Planning algorithms

## Tree method
1. Construct a search tree of thoughts guided by value estimations. 

1. Node of the tree is a partial solution $s=\left(x, z_{1:|s|}\right)$

1. Use a critic model $v$ to provide feedback $v\left(x, z_{1:|s|}\right)$

## Algorithmic method

1. Given past search history available at the $t$-th iteration $\left(s_{1}, v\left(s_{1}\right), \ldots, s_{t-1}, v\left(s_{t-1}\right)\right)$

1. a planning algorithm $\mathcal{A}$ iteratively determines the next search direction $\mathcal{A}\left(s_{t} \mid s_{1}, v\left(s_{1}\right), \ldots, s_{t-1}, v\left(s_{t-1}\right)\right)$

1.  and provides feedbacks for the current search progress $\mathcal{A}\left(v\left(s_{t}\right) \mid s_{1}, v\left(s_{1}\right), \ldots, s_{t}\right)$.

1. All information stored in the search tree used by the planning algorithm is flattened into the full context provided to the algorithm.
   1. Rather than explicitly constructing a search tree and implementing a planning algorithm
   1. Train a model to approximate this process.

1. This method enables the model to run an implicit search over the reasoning space directly via auto-regressive predictions.

1. Consequently, the model not only learns to solve a set of training problems but also develops the ability to tackle individual problems effectively, leading to improved generalization to unseen test problems.

## RL
1. Reward Model (RM), $r$ is a reward model that justifies the correctness of the proposed answer $y$ for the given problem $x$ based on the ground truth $y^{*}$, by assigning a value $r\left(x, y, y^{*}\right) \in\{0,1\}$.
1. For verifiable problems, the reward is directly determined by predefined criteria or rules. For example, in coding problems, we assess whether the answer passes the test cases. For problems with free-form ground truth, we train a reward model $r\left(x, y, y^{*}\right)$ that predicts if the answer matches the ground truth. Given a problem $x$, the model $\pi_{\theta}$ generates a CoT and the final answer through the sampling procedure $z \sim \pi_{\theta}(\cdot \mid x)$, $y \sim \pi_{\theta}(\cdot \mid x, z)$. The quality of the generated CoT is evaluated by whether it can lead to a correct final answer. In summary, we consider the following objective to optimize the policy

$$
\begin{equation*}
\max _{\theta} \mathbb{E}_{\left(x, y^{*}\right) \sim \mathcal{D},(y, z) \sim \pi_{\theta}}\left[r\left(x, y, y^{*}\right)\right] . \tag{1}
\end{equation*}
$$

By scaling up RL training, we aim to train a model that harnesses the strengths of both simple prompt-based CoT and planning-augmented CoT. The model still auto-regressively sample language sequence during inference, thereby circumventing the need for the complex parallelization required by advanced planning algorithms during deployment. However, a key distinction from simple prompt-based methods is that the model should not merely follow a series of reasoning steps. Instead, it should also learn critical planning skills including error identification, backtracking and solution refinement by leveraging the entire set of explored thoughts as contextual information.

### 2.3.2 Policy Optimization

We apply a variant of online policy mirror decent as our training algorithm (Abbasi-Yadkori et al. 2019, Mei et al. 2019, Tomar et al. 2020). The algorithm performs iteratively. At the $i$-th iteration, we use the current model $\pi_{\theta_{i}}$ as a reference model and optimize the following relative entropy regularized policy optimization problem,

$$
\begin{equation*}
\max _{\theta} \mathbb{E}_{\left(x, y^{*}\right) \sim \mathcal{D}}\left[\mathbb{E}_{(y, z) \sim \pi_{\theta}}\left[r\left(x, y, y^{*}\right)\right]-\tau \operatorname{KL}\left(\pi_{\theta}(x) \| \pi_{\theta_{i}}(x)\right)\right], \tag{2}
\end{equation*}
$$

where $\tau>0$ is a parameter controlling the degree of regularization. This objective has a closed form solution

$$
\pi^{*}(y, z \mid x)=\pi_{\theta_{i}}(y, z \mid x) \exp \left(r\left(x, y, y^{*}\right) / \tau\right) / Z .
$$

Here $Z=\sum_{y^{\prime}, z^{\prime}} \pi_{\theta_{i}}\left(y^{\prime}, z^{\prime} \mid x\right) \exp \left(r\left(x, y^{\prime}, y^{*}\right) / \tau\right)$ is the normalization factor. Taking logarithm of both sides we have for any ( $y, z$ ) the following constraint is satisfied, which allows us to leverage off-policy data during optimization

$$
r\left(x, y, y^{*}\right)-\tau \log Z=\tau \log \frac{\pi^{*}(y, z \mid x)}{\pi_{\theta_{i}}(y, z \mid x)}
$$

This motivates the following surrogate loss

$$
L(\theta)=\mathbb{E}_{\left(x, y^{*}\right) \sim \mathcal{D}}\left[\mathbb{E}_{(y, z) \sim \pi_{\theta_{i}}}\left[\left(r\left(x, y, y^{*}\right)-\tau \log Z-\tau \log \frac{\pi_{\theta}(y, z \mid x)}{\pi_{\theta_{i}}(y, z \mid x)}\right)^{2}\right]\right] .
$$

To approximate $\tau \log Z$, we use samples $\left(y_{1}, z_{1}\right), \ldots,\left(y_{k}, z_{k}\right) \sim \pi_{\theta_{i}}: \tau \log Z \approx \tau \log \frac{1}{k} \sum_{j=1}^{k} \exp \left(r\left(x, y_{j}, y^{*}\right) / \tau\right)$. We also find that using empirical mean of sampled rewards $\bar{r}=\operatorname{mean}\left(r\left(x, y_{1}, y^{*}\right), \ldots, r\left(x, y_{k}, y^{*}\right)\right)$ yields effective practical results. This is reasonable since $\tau \log Z$ approaches the expected reward under $\pi_{\theta_{i}}$ as $\tau \rightarrow \infty$. Finally, we conclude our learning algorithm by taking the gradient of surrogate loss. For each problem $x, k$ responses are sampled using the reference policy $\pi_{\theta_{i}}$, and the gradient is given by

$$
\begin{equation*}
\frac{1}{k} \sum_{j=1}^{k}\left(\nabla_{\theta} \log \pi_{\theta}\left(y_{j}, z_{j} \mid x\right)\left(r\left(x, y_{j}, y^{*}\right)-\bar{r}\right)-\frac{\tau}{2} \nabla_{\theta}\left(\log \frac{\pi_{\theta}\left(y_{j}, z_{j} \mid x\right)}{\pi_{\theta_{i}}\left(y_{j}, z_{j} \mid x\right)}\right)^{2}\right) . \tag{3}
\end{equation*}
$$

To those familiar with policy gradient methods, this gradient resembles the policy gradient of (2) using the mean of sampled rewards as the baseline (Kool et al. 2019, Ahmadian et al. 2024). The main differences are that the responses are sampled from $\pi_{\theta_{i}}$ rather than on-policy, and an $l_{2}$-regularization is applied. Thus we could see this as the natural extension of a usual on-policy regularized policy gradient algorithm to the off-policy case (Nachum et al. 2017). We sample a batch of problems from $\mathcal{D}$ and update the parameters to $\theta_{i+1}$, which subsequently serves as the reference policy for the next iteration. Since each iteration considers a different optimization problem due to the changing reference policy, we also reset the optimizer at the start of each iteration.
We exclude the value network in our training system which has also been exploited in previous studies (Ahmadian et al. 2024). While this design choice significantly improves training efficiency, we also hypothesize that the conventional use of value functions for credit assignment in classical RL may not be suitable for our context. Consider a scenario where the model has generated a partial $\mathrm{CoT}\left(z_{1}, z_{2}, \ldots, z_{t}\right)$ and there are two potential next reasoning steps: $z_{t+1}$ and $z_{t+1}^{\prime}$. Assume that $z_{t+1}$ directly leads to the correct answer, while $z_{t+1}^{\prime}$ contains some errors. If an oracle value function were accessible, it would indicate that $z_{t+1}$ preserves a higher value compared to $z_{t+1}^{\prime}$. According to the standard credit assignment principle, selecting $z_{t+1}^{\prime}$ would be penalized as it has a negative advantages relative to the current policy. However, exploring $z_{t+1}^{\prime}$ is extremely valuable for training the model to generate long CoT. By using the justification of the final answer derived from a long CoT as the reward signal, the model can learn the pattern of trial and error from taking $z_{t+1}^{\prime}$ as long as it successfully recovers and reaches the correct answer. The key takeaway from this example is that we should encourage the model to explore diverse reasoning paths to enhance its capability in solving complex problems. This exploratory approach generates a wealth of experience that supports the development of critical planning skills. Our primary goal is not confined to attaining high accuracy on training problems but focuses on equipping the model with effective problem-solving strategies, ultimately improving its performance on test problems.

### 2.3.3 Length Penalty

We observe an overthinking phenomenon that the model's response length significantly increases during RL training. Although this leads to better performance, an excessively lengthy reasoning process is costly during training and inference, and overthinking is often not preferred by humans. To address this issue, we introduce a length reward to restrain the rapid growth of token length, thereby improving the model's token efficiency. Given $k$ sampled responses
$\left(y_{1}, z_{1}\right), \ldots,\left(y_{k}, z_{k}\right)$ of problem $x$ with true answer $y^{*}$, let len $(i)$ be the length of $\left(y_{i}, z_{i}\right)$, min_len $=\min _{i}$ len $(i)$ and max_len $=\max _{i}$ len $(i)$. If max_len $=$ min_len, we set length reward zero for all responses, as they have the same length. Otherwise the length reward is given by

$
\text { lenreward }(\mathrm{i})=\left\{\begin{array}{rl}
\lambda & \text { If } r\left(x, y_{i}, y^{*}\right)=1 \\
\min (0, \lambda) & \text { If } r\left(x, y_{i}, y^{*}\right)=0
\end{array}, \quad \text { where } \lambda=0.5-\frac{\text { len }(i)-\text { minlen }}{\text { maxlen }- \text { minlen }} .\right.
$
