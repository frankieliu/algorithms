# Network Depth and Batch Size Scaling

## The Synergy Between Depth and Batch Size

A critical empirical finding in scaling deep RL networks is that **network depth and batch size must scale together**. Shallow networks lack the expressiveness required to effectively leverage the diverse information contained within large batches of transitions.

In deep Reinforcement Learning (RL), the benefit of scaling the batch size is obscured if the neural network models are too small. Scaled batch sizes only yield significant performance improvements when coupled with sufficiently large and expressive models.

## Why Shallow Networks Can't Use Large Batches

### Conventional Wisdom

The conventional wisdom observed in prior RL work was that simply increasing the batch size for traditional, often shallow, networks **"yields only marginal differences in performance"**.

### The Hypothesis

Researchers hypothesized that the **"small models traditionally used in RL may obscure the underlying benefits of larger batch size"**. This turned out to be correct—the issue wasn't with large batches themselves, but with the models being too limited to exploit them.

### The Evidence

Scaling the batch size **"becomes effective as network depth grows"**. This demonstrates that the increased capacity of deep networks is essential to realize the full benefit of sampling a broader set of data points in the batch.

## The Role of Expressiveness

The issue is that shallow networks lack the **expressive capacity** needed to process and differentiate between the diverse transitions in a large batch.

### What Expressiveness Enables

The primary function of increased expressiveness is to learn high-quality representations that enable the network to accurately differentiate between elements in the batch.

Deep networks learn **"richer representations"** that capture the complex topology of the environment, allowing them to:
- Distinguish between relevant and irrelevant state differences
- Separate positive examples from negative examples effectively
- Learn environment structure beyond simple heuristics

### Concrete Example: Maze Navigation

A clear demonstration of this comes from maze tasks:

#### Shallow Networks (Depth 4)
- Tend to **"naively rel[y] on Euclidean distance to the goal"** as a proxy for the Q-value
- Even when walls block the direct path, they use straight-line distance
- This shows a lack of representational capacity needed to accurately differentiate between states separated by a barrier
- They cannot effectively process the contrastive information in large batches

#### Deep Networks (Depth 64)
- Learn representations that **"effectively capture the topology of the maze"**
- Understand that a nearby state on the other side of a wall is actually far away in terms of path distance
- The expressive power of depth allows the network to distinguish relevant differences in the state space that shallow networks miss
- They can leverage large batches to learn these sophisticated distinctions

## Connection to Contrastive Learning

This finding is particularly relevant to **Contrastive RL (CRL)** and the **InfoNCE objective**:

### Why Depth Matters for InfoNCE

Recall that the InfoNCE objective requires the network to:
1. Identify positive goal-state-action associations (from the same trajectory)
2. Distinguish them from negative examples (from different trajectories)

**Shallow networks** processing a large batch:
- Cannot learn rich enough embeddings to make fine-grained distinctions
- Fall back on simple heuristics (like Euclidean distance)
- Fail to leverage the contrastive signal from negative examples

**Deep networks** processing a large batch:
- Learn high-dimensional embeddings that capture environment topology
- Can differentiate between subtle but important state differences
- Fully exploit the contrastive signal across many positive/negative pairs

### The Virtuous Cycle

This creates a virtuous cycle:
1. **Large batches** provide more diverse positive and negative examples
2. **Deep networks** have the capacity to learn from this diversity
3. **Better representations** emerge from processing contrastive examples
4. **Improved representations** enable better policy learning

Without sufficient depth, this cycle breaks down at step 2—the network simply can't process the information effectively.

## Practical Implications

### Scaling Strategy

When scaling RL systems:
- **Don't just increase batch size alone** - the network won't benefit
- **Don't just increase depth alone** - you need data diversity to justify the capacity
- **Scale them together** - depth provides capacity, batch size provides diversity

### Typical Parameters

In the CRL experiments that achieved 1024-layer scaling:
- **Batch size**: 512 transitions
- **Network depth**: Scaled from 4 to 1024 layers
- **Key finding**: Performance improvements from larger batches only materialized at greater depths

### Why This Matters for Self-Supervised RL

Self-supervised methods like CRL rely on learning from the **structure** of the data (trajectories, goals, state transitions) rather than sparse external rewards. This structure is revealed through:
- Comparing many different trajectories (large batches)
- Learning sophisticated representations (deep networks)

Shallow networks can't see the patterns; deep networks can. This is why self-supervised RL and depth scaling are fundamentally connected.

## Summary

The key insight: **Expressiveness (depth) unlocks batch scaling**.

- Shallow networks (<4 layers) + large batches = **wasted computation**
- Deep networks (64-1024 layers) + large batches = **rich representations**

In the context of Contrastive RL:
- The InfoNCE objective provides diverse contrastive examples via batches
- Deep networks provide the capacity to learn from these examples
- Together, they enable learning of environment topology and goal-directed behavior

This explains why traditional RL methods (shallow TD networks) couldn't benefit from large batches, while modern self-supervised methods (deep CRL networks) thrive on them.
