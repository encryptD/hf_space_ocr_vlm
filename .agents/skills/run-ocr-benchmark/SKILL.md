---
name: run-ocr-benchmark
description: Run OCR accuracy and latency benchmarks across all registered VLM backends and aggregation strategies
---

# Run OCR Benchmark

## When to use
Use this skill when evaluating OCR accuracy, latency, or throughput across backends or aggregation strategies.

## Instructions
1. Ask the user for the benchmark dataset path and which backends/strategies to include.
2. Ensure the benchmark script `scripts/benchmark.py` exists (create it if needed).
3. Run benchmarks with the following output format:
   - Per-backend character/word-level accuracy.
   - Per-aggregation strategy accuracy versus ground truth.
   - Latency percentiles (p50, p95, p99) per backend.
   - Throughput (images/second) for the full pipeline.
4. Save results to `results/benchmark_<timestamp>.json`.
5. Generate a summary markdown report at `results/benchmark_<timestamp>.md`.

## Important notes
- Use the existing ground-truth format (`data/ground_truth.jsonl`) if available.
- Run benchmarks with warm-up iterations to exclude model-loading latency.
- Ensure GPU memory is reset between backend runs if testing in a single process.

## Examples
- "Benchmark all backends on the synthetic OCR dataset"
- "Compare majority-vote vs. single-best aggregation accuracy"
