# SafetyDrift

**Predicting when AI agents cross the line before they actually do.**

When an LLM agent reads a confidential file, writes a summary, then emails it externally, no single step is unsafe, but the sequence is a data leak. We call this *safety drift*: individually safe actions compounding into violations.

Prior work has measured this problem. We predict it.

## The Idea
LLM agents take multi-step actions using tools (file access, code execution, network requests). Each step changes the agent's "safety state", a combination of what data it has seen, what capabilities it has used, and whether its actions can be undone. These states only escalate: once an agent reads credentials, that exposure never drops.

We model these safety state sequences as absorbing Markov chains. This lets us compute, at any point during execution, the probability that the agent will reach a safety violation within the next N steps. If that probability crosses a threshold, we intervene before the violation happens.

## What We Found
We ran 357 execution traces across 40 realistic agent tasks in four categories (data handling, system administration, research/communication, and code debugging).

**All agents eventually violate safety if left unsupervised.** Due to the monotonic nature of safety states, the absorption probability is 1.0 from every starting state. The question is not *if* but *when*.

**Points of no return are task dependent, not universal.** In communication tasks, an agent that reaches even a mild risk state has an 85% chance of violating safety within five steps. In technical tasks like sysadmin or code debugging, the probability stays below 5% from any state. This means a universal safety monitor is fundamentally suboptimal, and category aware monitoring is necessary.

**A lightweight monitor works.** A simple lookup table using category specific Markov models detects 94.7% of violations with 3.7 steps of advance warning, at negligible computational cost. This outperforms keyword matching (44.7% detection, 55.9% false positive rate) and even per step LLM judges (100% detection but 100% false positive rate and 779ms overhead per step).

## Paper
Dhodapkar, A. and Pishori, F. "SafetyDrift: Predicting When AI Agents Cross the Line Before They Actually Do." 2026.
