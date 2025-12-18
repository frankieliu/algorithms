# DSPy, GEPA, and ACE: Complete Guide

A comprehensive guide to understanding and integrating three powerful frameworks for building optimized LLM applications.

## Overview

This documentation is organized into four focused parts covering the theoretical foundations, practical implementations, and integration patterns of DSPy, GEPA, and ACE.

---

## Quick Reference

**[Acronyms & Quick Reference →](acronyms.md)** - Fast lookup for framework names, modules, data requirements, and metric types.

---

## Documentation Structure

### [Part 1: Framework Overview](01-framework-overview.md)
**Foundation concepts and integration architecture**

Learn about the three frameworks and how they work together:
- DSPy: The Structural Framework (Modularity & Reproducibility)
- GEPA: The Evolutionary Optimizer (Instruction Evolution)
- ACE: The Knowledge Manager (Strategic Memory)
- Integration patterns and the "Golden Stack"
- Framework independence and coupling

**Key Topics:**
- Visual diagrams of each framework's flow
- Where GEPA fits in the DSPy pipeline
- Combined flow through all three systems
- The roles: Skeleton (DSPy), Muscle (GEPA), Brain (ACE)

---

### [Part 2: DSPy Modules Deep Dive](02-dspy-modules.md)
**Understanding DSPy's thinking strategies**

Explore the three core DSPy modules that define how models process signatures:
- `dspy.Predict`: Direct responder for simple tasks
- `dspy.ChainOfThought`: Deliberative reasoning for complex logic
- `dspy.ReAct`: Agentic solver with tool integration
- How each module transforms your signatures
- GEPA's interaction with different modules

**Key Topics:**
- Signature transformation examples
- Internal prompt logic for each module
- Computational costs and trade-offs
- When to use each module type

---

### [Part 3: Training & Optimization](03-training-optimization.md)
**Data requirements and bootstrapping mechanisms**

Understand how DSPy learns and optimizes:
- What constitutes training data in DSPy (10-100 examples)
- The bootstrapping process: self-generating reasoning steps
- Why training data is mandatory for optimization
- How GEPA evolves beyond standard bootstrapping
- Data efficiency compared to traditional ML

**Key Topics:**
- Teacher-Student bootstrapping interaction
- The BootstrapFewShot optimizer
- Automating few-shot selection
- Generating reasoning traces
- Data amount guidelines (5-10, 20-50, 100+)

---

### [Part 4: Metrics & Validation](04-metrics-validation.md)
**Evaluation strategies and metric design**

Learn how to validate and score your LLM outputs:
- Labeled vs. Unlabeled data approaches
- When you need target answers (and when you don't)
- Metric design patterns: Code execution, constraints, LLM judges
- Cost vs. consistency trade-offs
- Why gold standards still matter

**Key Topics:**
- Three metric types: Exact match, Programmatic, AI-feedback
- Unlabeled metric examples (coding, summarization)
- The metric as "Environment" for optimization
- Practical metric implementation examples

---

## Quick Start

**If you're new to these frameworks:**
1. Start with [Part 1](01-framework-overview.md) to understand the big picture
2. Read [Part 2](02-dspy-modules.md) to learn how DSPy modules work
3. Study [Part 3](03-training-optimization.md) to prepare your training data
4. Review [Part 4](04-metrics-validation.md) to design your evaluation metrics

**If you're implementing a system:**
1. Define your task and choose a module type ([Part 2](02-dspy-modules.md))
2. Prepare 20-50 training examples ([Part 3](03-training-optimization.md))
3. Design your metric function ([Part 4](04-metrics-validation.md))
4. Understand the integration architecture ([Part 1](01-framework-overview.md))

---

## Framework Comparison at a Glance

| Framework | Primary Goal | What It Changes | Philosophy |
| --- | --- | --- | --- |
| **DSPy** | Modularity & Reproducibility | Code structure & Examples | LM as a Compiler |
| **GEPA** | Instruction Optimization | The text of the Prompt | LM as an Evolver |
| **ACE** | Strategic Memory | The Context/Knowledge provided | LM as a Knowledge Engineer |

---

## Key Concepts

### The Golden Stack
When used together, these frameworks create a complete optimization pipeline:
1. **ACE** retrieves the best strategy from past experience
2. **DSPy** provides the modular, reliable execution pipeline
3. **GEPA** ensures the instructions are optimally evolved

### Data Efficiency
Unlike traditional ML requiring thousands of examples:
- **5-10 examples:** Basic optimization possible
- **20-50 examples:** Optimal for GEPA evolution
- **100+ examples:** High-confidence compilation

### Module Complexity
- **Predict:** 1 LLM call, direct response
- **ChainOfThought:** 1 LLM call, includes reasoning
- **ReAct:** Multiple LLM calls, iterative with tools

---

## Original Document

The original comprehensive document is preserved as [optimization.md](optimization.md).

---

## Contributing

This documentation is designed to be modular and focused. Each part can be read independently or as part of the complete guide.

For questions or improvements, please refer to the specific section and provide context from the relevant part.
