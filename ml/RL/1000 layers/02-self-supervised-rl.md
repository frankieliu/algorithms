# Self-Supervised Reinforcement Learning

Self-supervised learning (SSL) and unsupervised goal-conditioned learning are key elements that, when combined, unlock the ability to successfully scale deep reinforcement learning (RL) networks.

## What is Self-Supervised Learning (SSL)?

Self-supervised learning (SSL) is a major learning paradigm that has driven breakthroughs in other machine learning fields, such as language and vision. The conventional wisdom suggests that large AI systems must be trained primarily in a self-supervised fashion, with RL used only for fine-tuning. SSL is a **key ingredient** needed if researchers hope to scale RL methods.

In the context of scaling research:

- **Mechanism:** Self-supervised learning in this domain (specifically, through Contrastive RL or CRL) involves framing the RL task as a **classification problem** rather than a traditional regression problem (like TD learning).
- **Objective:** CRL uses the **InfoNCE objective** (a generalization of cross-entropy loss) to train the critic. This classification-based approach is conjectured to be **more robust and stable** and exhibit better scaling properties than the regressive objectives used in standard TD methods.
- **Definition:** SSL and RL are seen not as diametric opposites, but as methods that **can be married together into self-supervised RL systems**.

## Relationship with Unsupervised Goal-Conditioned Learning

Unsupervised goal-conditioned learning describes the specific environment and task setting, while self-supervised learning (via CRL) is the necessary **algorithmic approach** used to succeed in that setting.

### The Unsupervised Setting
In this framework, the agent operates in an **unsupervised goal-conditioned setting** where **no demonstrations or external rewards are provided**. Instead, the agent must **explore (from scratch) and learn how to maximize the likelihood of reaching commanded goals**. The reward is typically sparse and binary, indicating only whether the goal proximity was reached.

### The Problem
Training large networks in this setting is difficult because the sparse, binary feedback means the system receives **very few bits of feedback relative to the number of parameters**, making learning highly unstable.

### The Solution (Self-Supervised RL)
To solve this challenge, the learning rule must be rethought, marrying RL and self-supervised learning into a system like Contrastive RL (CRL). The self-supervised nature of CRL allows the agent to learn effective policies that generalize across multiple goals **without relying on the sparse external reward function** or demonstrations. Instead, the InfoNCE objective generates its learning signal internally, typically by treating achieved states in a trajectory as positive goals that could have been commanded.

In short, **self-supervised RL** is the required self-contained learning paradigm that overcomes the fundamental challenge of **sparse feedback** inherent in the **unsupervised goal-conditioned setting**, enabling performance gains through network scaling.

## How Self-Supervised Learning Replaces the Reward Need

The term "self-supervised RL" (specifically Contrastive RL or CRL) is the **mechanism** designed to succeed despite the sparsity of the goal-conditioned reward function.

Instead of relying heavily on the external reward $r_g$, self-supervised RL uses the goals **internally** to generate its own dense learning signal:

- **Goal as Classification Label:** CRL frames the RL problem as a **classification task** using the **InfoNCE objective**. The critic, $f_{\phi,\psi}(s, a, g)$, measures the distance between the state-action embedding and the goal embedding.
- **Internal Learning Signal:** The agent is trained to maximize this critic output. This classification approach effectively learns whether a state-action pair leads toward a goal state, relying on the **structure provided by the goal space** and the achieved states within trajectories to provide meaningful feedback, thereby **bypassing the need for a dense external reward**.

Therefore, while a formal reward function ($r_g$) technically exists in the MDP definition, the learning success of these "unsupervised" or "self-supervised" methods depends on **not relying on that sparse reward signal** but instead on the goals and self-generated contrastive objectives.

## SSRL as the Task Environment Solution

The relationship is described not as one achieving the other, but rather the specialized **Self-Supervised RL (SSRL) algorithm** being applied to the challenging **Unsupervised Goal-Conditioned RL environment** to achieve breakthroughs in scaling.

1. **Unsupervised Goal-Conditioned RL: The Task Environment**
   The term "**unsupervised goal-conditioned setting**" describes the specific training setup in which the experiments are conducted. In this environment, the agent operates under the severe constraints of a multi-goal RL problem where **no demonstrations or external rewards are provided**, forcing the agent to explore from scratch and learn how to maximize the likelihood of reaching commanded goals.

2. **Self-Supervised RL (SSRL): The Algorithm**
   **Self-Supervised RL** is the category of algorithms designed to solve this difficult unsupervised problem. The researchers propose that reinforcement learning and self-supervised learning are **not diametric learning rules**, but rather can be "**married together into self-supervised RL systems**".

3. **The Specific Mechanism (CRL)**
   The specific algorithm used is **Contrastive RL (CRL)**, which is an example of an SSRL system. The effectiveness of CRL stems from its self-supervised nature, as it uses the **InfoNCE objective** (a generalization of cross-entropy loss) to frame the value-learning problem as a **classification task**. This classification-based approach yields a more robust and stable learning signal, which is necessary to overcome the sparsity of feedback inherent in the unsupervised goal-conditioned setting.

In summary, the success of scaling deep networks (up to 1024 layers) happens because the **Self-Supervised Contrastive RL (CRL) algorithm** provides the necessary architectural stability and dense internal signal required to effectively learn in the challenging **unsupervised goal-conditioned setting**.
