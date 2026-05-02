---
name: billing-optimization-plan
description: Optimize model-credit usage by selecting the lightest model tier that can still solve the request. Use this skill at the start of every interaction, even if the user does not mention billing, to avoid unnecessary expensive-model usage.
---

# Billing Optimization Plan

## When to use
Use this skill for every interaction before doing planning, code edits, or tool execution.

## Core policy
If a task does not require deep reasoning, planning, or multi-step problem solving, choose a lightweight model tier to reduce credit usage.

## Model tiers
- `lightweight`: Simple, direct requests with one clear step.
- `balanced`: Moderate requests with a few steps but low ambiguity.
- `deep`: Complex, ambiguous, or high-risk requests that need careful reasoning and planning.

## Model candidates by tier
Use the closest available model from each class when mapping tiers to concrete models:

- `lightweight` (fast + low cost)
  - Claude Haiku class (example: `claude-3-5-haiku`)
  - GPT mini class (example: `gpt-4.1-mini`)
  - Small instruct open models (example: `llama-3.1-8b-instruct`)

- `balanced` (mid-cost + stronger general reasoning)
  - Claude Sonnet class (example: `claude-sonnet-4-5`)
  - GPT standard class (example: `gpt-4.1`)
  - Mid-size instruct open models (example: `qwen2.5-32b-instruct`)

- `deep` (highest reasoning quality)
  - Claude Opus class (example: `claude-opus-4-1`)
  - Reasoning-focused models (example: `o3`)
  - Frontier open reasoning models (example: `deepseek-r1`)

If a listed model is unavailable, pick the nearest equivalent in the same tier.

## Classification rubric
Choose `lightweight` when most of these are true:
- The user asks for a command, definition, tiny edit, or short transformation.
- There is little or no ambiguity in the request.
- The likely answer does not require decomposition into multiple dependent steps.

Choose `balanced` when one or more of these are true:
- The task has 2-4 steps but is still straightforward.
- The request needs small comparisons, tradeoffs, or moderate code changes.
- The response benefits from brief reasoning but not deep architectural analysis.

Choose `deep` when one or more of these are true:
- The task requires planning, architecture, migration strategy, or root-cause investigation.
- The task is high-impact (security, data integrity, production-risky changes).
- The task involves many dependencies, unclear requirements, or broad refactoring.

## Interaction procedure (apply every time)
1. Quickly classify the request as `lightweight`, `balanced`, or `deep`.
2. Start with the lowest tier that can credibly solve the request.
3. Escalate only when blocked by complexity, ambiguity, or quality risk.
4. Do not pre-escalate solely because a larger model is available.

## Escalation and de-escalation rules
- Escalate `lightweight → balanced` if the task expands into multiple dependent steps.
- Escalate `balanced → deep` if uncertainty remains high after initial decomposition.
- De-escalate when the request narrows to a simple follow-up.

## Benchmarking
Run the local benchmark harness to validate decision quality and estimated savings:

`python .agents/skills/billing-optimization-plan/scripts/benchmark.py`

This benchmark reports:
- Tier classification accuracy against curated prompts.
- Unsafe downgrade rate (complex prompts classified too low).
- Estimated credit savings versus an `always_deep` baseline.
