# Part 3: Training & Optimization

> **Navigation:** [← Part 2: DSPy Modules](02-dspy-modules.md) | [Part 4: Metrics & Validation →](04-metrics-validation.md)

## Table of Contents
- [3.1 Training Data in DSPy](#31-training-data-in-dspy)
- [3.2 Bootstrapping Process](#32-bootstrapping-process)
- [3.3 Why Training Data is Mandatory](#33-why-training-data-is-mandatory)

---

## 3.1 Training Data in DSPy

In traditional Machine Learning, training data consists of thousands of labeled examples. In **DSPy**, "training data" is much lighter and serves a different purpose.

### What is "Training Data" in DSPy?

You don't need a massive dataset. DSPy typically works with:

* **Small Sample Size:** Usually between **10 and 100 examples**.
* **Example Structure:** A set of `dspy.Example` objects containing the inputs (e.g., a question) and the target outputs (e.g., the answer).
* **Labels:** You only need to label the *final* output. You do **not** need to write out the "Chain of Thought" or the "Intermediate Steps"—DSPy will figure those out for you.

#### Example: A Simple DSPy Training Set

```python
# A simple DSPy Training Set
trainset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs('question'),
    dspy.Example(question="What is 2+2?", answer="4").with_inputs('question')
]
```

---

## 3.2 Bootstrapping Process

### What is "Bootstrapping"?

This is the core "learning" mechanism. Since your training data only has the *Question* and the *Answer*, the model doesn't know what the "Reasoning" (Chain of Thought) should look like.

**Bootstrapping is the process where DSPy "self-generates" the intermediate reasoning steps.**

### How the Bootstrap Process Works

1. **Teacher-Student Interaction:** DSPy uses a "Teacher" (usually a more powerful model like GPT-4) to try and solve your training examples using your program's structure.
2. **Trace Collection:** If the Teacher gets the answer right (matches your label), DSPy saves the *entire trace* (the reasoning, the search steps, and the final answer).
3. **Few-Shot Assembly:** These successful "traces" are then compiled into a highly optimized prompt that is given to the "Student" (your actual production model).

---

### The "BootstrapFewShot" Optimizer

The most common way to do this is using the `BootstrapFewShot` optimizer. It essentially "searches" for the best examples to put into your prompt.

* **Standard Few-Shot:** You manually pick 3 examples and hope they work.
* **DSPy Bootstrapping:** DSPy tries 50 examples, sees which 5 produced the most accurate "Chain of Thought" logic, and uses those as the permanent prompt.

---

### How GEPA Evolves Bootstrapping

While `BootstrapFewShot` just picks the best **examples** (the data), **GEPA** takes it a step further. GEPA looks at the "failed" bootstrapped traces and says:

> "The model failed these examples because the instructions were too vague. Let me mutate the **Instruction Text** and try bootstrapping again."

---

### Summary: Data vs. Bootstrap

| Feature | Training Data | Bootstrapping |
| --- | --- | --- |
| **Input** | Question + Answer pairs | Training Data + Program Logic |
| **Output** | A static list of examples | A "Compiled" prompt with reasoning |
| **Effort** | Low (You just provide the final answer) | High (DSPy runs multiple LLM calls to find success) |
| **Goal** | To define the target | To find the best "path" to that target |

---

## 3.3 Why Training Data is Mandatory

In traditional prompting, you are the "optimizer"—you look at a model's mistake, manually tweak a sentence in the prompt, and try again.

In **DSPy**, the machine is the optimizer. For a machine to optimize a program, it needs a **success criterion** and **examples** of what "good" looks like. Providing training data allows DSPy to move from "guessing" to "compiling."

Here are the three specific reasons why training data is mandatory for the DSPy/GEPA/ACE stack:

---

### Reason 1: To Automate "Few-Shot" Selection

LLMs perform significantly better when given 3–5 examples of a task (Few-Shot).

* **Without Data:** You have to manually write these examples, which is tedious and often biased.
* **With Data:** DSPy's optimizers (like `BootstrapFewShot`) search through your training data to find the most effective examples to "plug in" to your prompt. It tries different combinations to see which specific examples lead to the highest accuracy.

---

### Reason 2: To Generate "Reasoning Traces" (The Rationale)

As we discussed with `ChainOfThought`, the model needs to "think" before it answers.

* DSPy uses your training data to perform **Bootstrapping**.
* It takes a question from your data, tries to solve it, and if it gets the answer right, it **saves the reasoning steps** it used.
* These saved steps become the "gold standard" reasoning examples that guide the model in the future.

---

### Reason 3: To Provide a Feedback Signal for GEPA

If you are using **GEPA**, the training data acts as the "Environment" for evolution.

* GEPA generates multiple versions of a prompt (Mutations).
* To know which mutation is the "fittest," it must test them against the **Training Data**.
* Without data, GEPA has no way to calculate a **Pareto Frontier** or determine which instructions actually improved performance.

---

### How Much Data Do You Actually Need?

Unlike Deep Learning, which requires thousands of rows, DSPy is "Data-Efficient."

| Data Amount | Result |
| --- | --- |
| **0 Examples** | You are just doing standard "Zero-Shot" prompting (No optimization possible). |
| **5–10 Examples** | Enough for **BootstrapFewShot** to find a few good examples. |
| **20–50 Examples** | The "Sweet Spot" for **GEPA** to evolve high-quality instructions. |
| **100+ Examples** | Allows for rigorous testing and high-confidence "Compilation." |

---

### The "Metric": The Partner to Your Data

Providing data is only half the battle. You also provide a **Metric function**. The training data provides the *input*, and the metric provides the *score*.

#### Example Metric Function

```python
def validate_answer(example, pred, trace=None):
    # Does the predicted answer match the gold label in our training data?
    return example.answer.lower() == pred.answer.lower()
```

This metric works together with your training data during the `compile()` process to determine which prompts and examples lead to the best performance.

---

> **Navigation:** [← Part 2: DSPy Modules](02-dspy-modules.md) | [Part 4: Metrics & Validation →](04-metrics-validation.md)
