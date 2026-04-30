---
name: add-aggregation-strategy
description: Add a new text aggregation or voting strategy for combining OCR outputs from multiple VLM backends
---

# Add Aggregation Strategy

## When to use
Use this skill when adding a new strategy for combining or voting on OCR text outputs from multiple VLM backends.

## Instructions
1. Ask the user for the strategy name and a description of the algorithm (e.g., majority vote, confidence-weighted average, LLM-based reconciliation).
2. Create a new module under `src/aggregation/<strategy_name>.py` implementing the standard `BaseAggregator` interface.
3. Register the strategy in `src/aggregation/__init__.py`.
4. Add configuration schema to `config/aggregation.yaml`.
5. Add a unit test under `tests/aggregation/test_<strategy_name>.py`.
6. Update `README.md` with the new strategy.

## Important notes
- Aggregators receive a list of `(backend_name, text_output, confidence_score)` tuples.
- Return a single `(aggregated_text, confidence_score, metadata)` result.
- Keep the strategy deterministic and unit-testable without external API calls.

## Examples
- "Add a simple majority-vote strategy for OCR text"
- "Create a confidence-weighted merge strategy"
