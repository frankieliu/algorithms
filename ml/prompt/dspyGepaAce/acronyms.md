# Acronyms & Quick Reference

> **Navigation:** [Main Documentation →](README.md)

## The Three Frameworks

| Acronym | Full Name | Role | Analogy | What It Optimizes |
|---------|-----------|------|---------|-------------------|
| **DSPy** | Declarative Self-improving Python | The Skeleton | The structure/blueprint of the machine | Code structure & Examples |
| **GEPA** | Genetic Evolution of Prompts with Alternatives | The Muscle | The fine-tuning process (tightening bolts, adjusting fuel) | The text of the Prompt/Instructions |
| **ACE** | Agentic Context Engineering | The Brain | The library/manual for problem-solving | The Context/Knowledge provided |

---

## Quick Comparison

### What Each Framework Does

| Framework | Primary Function | Can Work Alone? | Typical Data Needed |
|-----------|-----------------|-----------------|---------------------|
| **DSPy** | Structures LLM programs into modular signatures | Yes | 5-100 examples |
| **GEPA** | Evolves instruction text through genetic algorithms | No (needs DSPy or similar) | 20-50 examples |
| **ACE** | Manages strategic memory and past experiences | Yes | Ongoing task history |

---

## The Golden Stack

When combined, they form a complete optimization pipeline:

```
User Request
    ↓
[ACE: Retrieve best strategy from Playbook]
    ↓
[DSPy: Execute modular program with proper structure]
    ↓
[GEPA: Use evolved instructions for optimal results]
    ↓
Result + Feedback → ACE Playbook
```

---

## DSPy Modules Quick Reference

| Module | Complexity | LLM Calls | Best For |
|--------|-----------|-----------|----------|
| `Predict` | Low | 1 | Simple classification, extraction |
| `ChainOfThought` | Medium | 1 (longer) | Logic, math, reasoning |
| `ReAct` | High | Multiple | Research, tool use, multi-step |

---

## Data Requirements

| Amount | What You Can Do |
|--------|-----------------|
| 0 examples | Zero-shot only (no optimization) |
| 5-10 examples | Basic BootstrapFewShot |
| 20-50 examples | **Optimal for GEPA** |
| 100+ examples | High-confidence compilation |

---

## Metric Types

| Type | Needs Target? | Example Use Case | Cost |
|------|---------------|------------------|------|
| **Labeled** | Yes | QA, classification | Free |
| **Programmatic** | No | Code validation, formatting | Free |
| **LLM-as-Judge** | No | Creative writing, tone | Expensive |

---

## Related Concepts

- **Signature**: The contract defining inputs and outputs (e.g., `question -> answer`)
- **Bootstrapping**: Auto-generating reasoning traces from successful examples
- **Pareto Frontier**: Set of diverse, high-performing prompt variants in GEPA
- **Playbook**: ACE's memory of successful strategies for different tasks
- **Teleprompter**: Original DSPy term for optimizers (now just "Optimizer")
- **Metric**: Function that scores whether a prediction is successful

---

> **Navigation:** [Main Documentation →](README.md) | [Part 1: Framework Overview →](01-framework-overview.md)