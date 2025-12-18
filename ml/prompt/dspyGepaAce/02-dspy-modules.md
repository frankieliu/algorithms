# Part 2: DSPy Modules Deep Dive

> **Navigation:** [← Part 1: Framework Overview](01-framework-overview.md) | [Part 3: Training & Optimization →](03-training-optimization.md)

## Table of Contents
- [2.1 Module Types](#21-module-types)
- [2.2 Signature Transformations](#22-signature-transformations)

---

## 2.1 Module Types

In DSPy, these three modules define **how** the model processes your Signature. While the Signature defines the "contract" (Inputs and Outputs), these modules define the "thinking strategy."

Think of them as different levels of mental effort for the model.

---

### `dspy.Predict` (The Direct Responder)

This is the most basic module. It takes the input and asks the model for the output immediately, without any intermediate steps.

* **Behavior:** Direct Input → Output.
* **Prompting Style:** Standard zero-shot or few-shot instruction.
* **Best For:** Simple, deterministic tasks like sentiment analysis, translation, or entity extraction.
* **Analogy:** Asking someone a trivia question where they either know the answer or they don't.

---

### `dspy.ChainOfThought` (The Deliberator)

This module automatically modifies your Signature to include a `rationale` (reasoning) field before the final answer.

* **Behavior:** Input → **Reasoning** → Output.
* **Prompting Style:** It injects a "Let's think step-by-step" instruction into the background prompt.
* **Best For:** Tasks requiring logic, math, or multi-step common sense. Even if you didn't define a "reasoning" field in your Signature, `ChainOfThought` will add one and return it in the result.
* **Analogy:** Asking someone to solve a math word problem and requiring them to show their work.

---

### `dspy.ReAct` (The Agentic Solver)

"ReAct" stands for **Reasoning + Acting**. This is the most complex module and turns your Signature into an iterative agent loop.

* **Behavior:** Input → [ **Thought** → **Action** (Tool Call) → **Observation** ] (Repeat) → Final Answer.
* **Prompting Style:** It forces the model to use a loop: it thinks about what to do, calls a tool (like a search engine or calculator), looks at the result, and repeats until it has the answer.
* **Best For:** Complex tasks that require external data or multiple steps of information gathering (e.g., "Find the age of the current President's spouse and calculate their birth year").
* **Analogy:** Giving someone a research task and a laptop to look things up until they are finished.

---

### Comparison Summary

| Module | Strategy | Output Fields | External Tools? | Complexity |
| --- | --- | --- | --- | --- |
| **`Predict`** | Direct | Just your Signature fields | No | Low |
| **`ChainOfThought`** | Step-by-step | Signature + **`rationale`** | No | Medium |
| **`ReAct`** | Loop | Signature + **`trace`** | **Yes** | High |

---

### Why this matters for GEPA and ACE

* **GEPA** is most powerful when used with **`ChainOfThought`** or **`ReAct`**. Since GEPA optimizes by *reflecting* on failures, it needs to see the model's "thinking" to suggest better instructions.
* **ACE** often manages the "tools" and "past strategies" that a **`ReAct`** agent uses to decide which Action to take next.

---

## 2.2 Signature Transformations

Let's take one simple Signature and look at how DSPy transforms it under the hood for each module.

### The Base Signature

```python
class SimpleQA(dspy.Signature):
    """Answer the question based on facts."""
    question = dspy.InputField()
    answer = dspy.OutputField()
```

---

### Transform 1: `dspy.Predict(SimpleQA)`

This is the "Zero-Complexity" version. It translates the signature directly into a prompt template.

**The Internal Prompt Logic:**

> **Context:** Answer the question based on facts.
> **Question:** {question}
> **Answer:** [Model generates answer here]

* **Result:** You get just the answer.
* **Failure Mode:** If the question is hard, the model might "hallucinate" or guess because it wasn't forced to think first.

---

### Transform 2: `dspy.ChainOfThought(SimpleQA)`

DSPy **intercepts** your signature and dynamically adds a `rationale` field. It tells the model that the reasoning is a prerequisite for the answer.

**The Internal Prompt Logic:**

> **Context:** Answer the question based on facts.
> **Question:** {question}
> **Reasoning:** Let's think step by step. [Model generates reasoning here]
> **Answer:** [Model generates answer here]

* **Result:** The output object contains `prediction.rationale` and `prediction.answer`.
* **Why it's better:** By verbalizing the logic, the model's accuracy on complex tasks increases significantly.

---

### Transform 3: `dspy.ReAct(SimpleQA, tools=[search])`

This is the "Iterative" version. It turns the signature into a loop. It doesn't just add a field; it changes the model's entire operational mode into **Thought → Action → Observation.**

**The Internal Prompt Logic (Simplified Loop):**

> **Task:** Answer the question based on facts.
> **Question:** {question}
> **Thought 1:** I need to find the current height of the Eiffel Tower.
> **Action 1:** Search[Eiffel Tower height 2025]
> **Observation 1:** The Eiffel Tower is 330 meters tall after a new antenna was added.
> **Thought 2:** I have the information needed.
> **Answer:** 330 meters.

* **Result:** The output object contains the full `trace` of the thoughts and actions.
* **Why it's better:** It can correct itself. If the "Observation" reveals the first "Thought" was wrong, the agent can pivot.

---

### Side-by-Side Structural Comparison

| Feature | `Predict` | `ChainOfThought` | `ReAct` |
| --- | --- | --- | --- |
| **Logic Type** | Linear / Instant | Linear / Deliberative | Recursive / Agentic |
| **Field Injection** | None | Adds `rationale` | Adds `thought`, `action`, `observation` |
| **Compute Cost** | Lowest (1 LLM call) | Low (1 LLM call, more tokens) | High (Multiple LLM calls) |
| **Best Used With** | Basic Classification | Logical Reasoning | External Research/Tools |

---

### How GEPA interacts with these modules

If you use **GEPA** to optimize these:

* In `Predict`, GEPA evolves the **instruction text**.
* In `ChainOfThought`, GEPA evolves the **reasoning style** (e.g., "Think like a scientist").
* In `ReAct`, GEPA evolves the **decision-making strategy** (e.g., "Search twice before concluding").

---

> **Navigation:** [← Part 1: Framework Overview](01-framework-overview.md) | [Part 3: Training & Optimization →](03-training-optimization.md)
