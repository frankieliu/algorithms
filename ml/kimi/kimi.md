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