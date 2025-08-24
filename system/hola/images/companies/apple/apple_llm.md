# Foundational LLM Questions

1. Difference btw pre-training and post-training
   - pre-training (unsupervised) language patters
   - post-training (fine-tuning / alignment)
     - adapts model to specific tasks or human preferences

1. Transformer architecture?
   - self attention

1. Impact of scaling?
   - larger model/data/compute improve performance predictably (scaling laws) - diminishing returns / higher costs

1. Challenges of llm
   - compute cost
   - env impact
   - over fitting
   - alignment risk (bias, hallucination)

# Fine Tuning and Adaptation

1. SFT?
   - fine tuning on labeled task-specific data (eg instruction-following)

1. PEFT vs full-tuning
   - PEFT updates only smaller adapter layers, reducing memory cost
   - full tuning adjusts all parameters

1. LoRA
   - freeze base mode, inject low-rank matrices to adapt weights efficiently

1. prompt tuning
   - learns soft prompts (continuous embeddings) instead of model weights

# RLHF

1. Why
   - align model output to human preferences via reward modeling + RL

1. what are the steps
   a. collect ranking
   b. train reward model
   c. optimize llm policy PRO using reward

1. limitations
   - costly
   - prone to reward hacking
   - alternative DPO, RLAIF

1. PPO
   - policy gradient method updates LLM, preventing unstable, large wt changes

# Post-training optimization

1. Quantization
   - reduce model weight precision, eg. 32 bit -> 4 bit

1. Distillation
   - train smaller student model to mimic a larger teacher model

1. Speculative decoding
   - use cheaper draft model to  propose tokens
   - main model verifies

1. Reduce hallucinations
   - use RAG, fine-tune on factual data, apply contrastive decoding

# Evaluation and Alignment

1. How to evaluate LLMs
   - benchmarks (MMLU, HELM)
   - human eval
   - metrics for coherence, toxicity, and factuality

1. Truthfulness
   - train with factual data, use verification tools, or prompt with chain of thought to improve reasoning

1. CoT
   - encourage step by step reasoning in prompts to boost complex task performance

1. Mitigate bias
   - curate diverse training data, apply debiasing algorithms and use RLHF with fairness rewards

# Advanced Topics

1. MoE
   - split model into sub-networks activated per input, improving efficiency (Mistral, Switch Transformer)

1. RAG vs SFT
   - RAG retrieves external knowledge dynamically
   - SFT embeds knowledge into weights
   - Use RAG for dynamic data

1. Open vs closed weights
   - Open weights LLaMA allow customization
   - Closed offer polish but less control

# Practical/Scenario-Based

1. Overfitting in fine-tuning
   - reduce learning rate
   - add dropout
   - use early stopping/LoRA for parameter efficiency

1. Deploy 70B model on edge
   - quantize GGUF 4-bit
   - MoE
   - offload layers to CPU/disk

1. Harmful content fixes
   - apply RLHF, content filters, or fine-tune on safety-focused datasets

1. Domain alignment pipeline
   - SFT on domain data
   - RLHF with domain experts
   - RAG for real-time knowledge

# Overfitting

1. What is it
   - LLM memorized the FT data instead of generalizing
   - Poor perfomance on OOD data
   - Degraded baseline capabilities

1. Why does it happen
   - small dataset
   - too many epochs on same data
   - high model capacity
   - task-noise mismatch, fine-tuning data is noisy or misaligned with target task

1. How to detect
   - validation metrics
     - monitor loss/accuracy on validation if loss drops during training but rises on validation
     - generalization tests: evaluate on OOD tasks, benchmarks MMLU
     - qualitative checks: manual inspection for regurgitation of training examples

1. Mitigation

   1. Data level
      - increase dataset size
      - data augmentation
      - balance tasks

   1. Training level
      - early stopping when validation plateaus
      - regularization
        - dropout (0.1)
        - weight decay penalize large weights
      - reduce LR (1e-5 to 1e-6)
   1. PEFT
      - LoRA
      - Adapter layers
   1. Evaluation and debugging
      - Checkpointing
        - save multiple model versions and compare validation performance
        - gradient clipping prevent exploding gradients (common in small datasets)     
# Trade-offs to consider
  - Underfitting vs overfitting
    - too little model unadapted
    - too much causes memorization
  - Task specificity
    - Over optimizing for one task may harm zero-shot capabilities
# Example Workflow to avoid Overfitting
  1. Start with large diverse dataset
  1. Use LoRA + low LR (2e-5) + dropout (0.1)
  1. Evaluate on validation set every 500 steps, stop early if loss rises
  1. Test on OOD tasks (unseen instructions or public benchmarks)

# MMLU 
1. Massive Multitask Language Understanding
   - broad many subjects
   - general knowledge
   - requires reasoning
   - standardized leaderboard
   - Limitations: multiple choice
# HELM
1. Holistic Evaluation of LMs
   - 42 scenarios (summariation, QA, bias detection)
   - multiple metrics:
     - Robustness (typos)
     - Accuracy
     - Fairness (demographics bias)
     - Efficiency (latency and memory usage)
   - real-world conditions, adversarial inputs and long-tail queries



# Model drift
1. Data drift

   1. KL divergence / Population Stability Index
   - compare input/output distributions
   - bin the data (eg input text length, topic frequency)
   - look as percentage of total per bin 
     (cur - pre) * log(cur/pre)

   1. Chi Square Test: shift in categorical data
   - X^2 = sum (Observed_count - Expected_count)^2/Expected_count
   - degree of freedom: number of categories - 1
   - critical value: look up X^2 threshold for df and significance
     level (eg 5% significance and df=2 => X^2 = 5.99)
   - if above critical level, drift is detected

   Chi-square detects if Drift exists
   PSI quantifies magnitude

   1. Embedding drift: track changes in vector-space representation of inputs
1. Concept drift
   1. User Feedback
   1. Output anomalies (refusals, hallucination, repetition)
1. Performance drift
   1. MMLU / HELM doing worse
   1. A/B Testing : compare new vs old model versions on live traffic
   1. Shadow Mode: log differences from production outputs

1. Mitigation
   - continuous fine-tuing
   - retraining triggers
   - modular updates
1. Workflow
   - log 1% of daily inputs/outputs
   - compute embedding drift weekly
   - retrain if MMLU scores drop by >5%

# Embedding drift
1. Change in vector representation of inputs or outputs
1. Caused by
   - data distribution shift
   - model updates
   - contextual changes
1. Measure
   1. cosine similarity
   1. Statistical test
      - maximum mean discrepancy (MMD)
      - PCA / Jensen-Shannon Divergence
   1. Cluster based
1. Mitigation
   - dynamic fine tuning
   - RAG - fetch relevant embeddings from an up-to-date DB
1. Importance
   - drift degrades accuracy in downstream task (search / classification)
   - user experience - misrepresent queries
   - bias amplification

# details
1. question about competing ideas
1. question about past project
1. question koko bananas
1. question on transformer, expedite inference, layer norm





# Gemini

Andy Tseng

1. Interesting / challenging problems 
1. Foundational research / practical problems
1. Collaboration with other teams
1. Collaborative vs individual-driven
1. Success in the role 90/long term
1. ICT4 ICT5 differences
1. Career progression
1. Setbacks or failed projects

Personal views
Siri - flexibility to reach the goal
Collaboration - different projects
- nature of the project
- can't do it by yourself
- structure / other tools
- complex system
  - so many
  - have access to everything
  - area about search
    - global knowledge / contacts local
    - help on both
    - internal tool to siri
    - heavily collaborate with them
    - mutually respectable accross teams
- failure taken
  - post ponent how to 
  - failure take as a lesson disappointment
  - fighting for a year
  - post ponent leader take responsibility
  - see what what is the real problem and 
    - fix the 
    - big delay
  - quite different than Meta
    - see failure is quite normal
    - value user experience the most
    - 13-14 year - don't want something
    - million of requests
      - quality of next generation of project
      - user will be satisfied
      - cancel internally
      - disappointing 
      - engineers is hard
    - product
      - dedicated team / make spec for the product
      - they know a lot a ux and interface
      - communicate what siri communicates / constraining
        - all LLMs
        - everything can be rendered - underlying technology
          - risk
        - HI team 

## Day to day
1. Teams approach to technical debt
2. Handle code reviews, testing, and other quality assurance processes
3. Collaboration tools / project management: Jira / workflow / how to track progress
   1. complete process - subproject taskboard
   2. development for LM is so fast
      1. later on new stuff - ok to bottleneck overtaking
      2. adaptive - taking things very fast
      3. boards stick to old system things are moving very fast
      4. PM there are so many resources
4. On-call rotations and bug triaging, after hours, remote teams

# Personal growth
1. how do you get feedback on the team
2. continuous feedback / frequency of reviews
   - constructive criticism
3. opportunities to learn new skills / different areas of AI

   - reflecting on this everyday
   - publication and doing research
   - prod - good intern and fully research on intern
   - 2022 apple all because of interns
     - flexibility in this team
     - good reputations
     - interm tier 1 conference
     - main tutor - pure researhc visual LLM multimodal
   - daily growth / career growth
     - might be super familiar you are good
     - saturated in the growth curve
     - find a balance new things and things that you are good at
     - challenge / propose new things
   - manager Hong - allows to choose boundary / flexibility
     - choose we are most interesting in
     - explore
     - usually propose
       - proof of concept - daily work

4. celebrate success both large and small
5. team outing

# Broader impact
1. how does the team think about the impact of its work
2. ethical considerations - public scrutiny over AI
3. in perfect world, what would this team be able to achieve in the next 5-10 years

# Track
1. successful query resolution rate
2. time to first response
3. user engagement with suggested content
4. model drift and degradation over time
5. what process for A/B testing or live experimentation of new models
6. how to balance achieving high metrics with other constraints like latency, comp cost, data private
 
# feedback

innovation team
foundation team
part of their scope

Siri product team
- ICT5 - significant transparent with cross functional team
- ICT5 - technical side

# team summary
1. buy some time
2. schedule conversation
3. send team summary - try to get soe
