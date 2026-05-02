#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


TIERS = ("lightweight", "balanced", "deep")
TIER_TO_INDEX = {tier: i for i, tier in enumerate(TIERS)}
TIER_COST = {"lightweight": 1, "balanced": 3, "deep": 8}

DEEP_SIGNALS = (
    "plan",
    "strategy",
    "architecture",
    "root cause",
    "investigate",
    "migration",
    "refactor",
    "multi-tenant",
    "reliability",
    "rollout",
)
STRONG_DEEP_SIGNALS = (
    "refactor",
    "root cause",
    "architecture",
    "multi-tenant",
    "migration plan",
    "implementation plan",
)

BALANCED_SIGNALS = (
    "compare",
    "tradeoff",
    "update",
    "add",
    "script",
    "test",
    "verify",
    "checklist",
    "dockerfile",
)


def classify_prompt(prompt: str) -> str:
    text = prompt.lower()
    if any(signal in text for signal in STRONG_DEEP_SIGNALS):
        return "deep"
    deep_hits = sum(signal in text for signal in DEEP_SIGNALS)
    balanced_hits = sum(signal in text for signal in BALANCED_SIGNALS)

    if deep_hits >= 2:
        return "deep"
    if deep_hits == 1 and balanced_hits >= 1:
        return "deep"
    if balanced_hits >= 1:
        return "balanced"
    return "lightweight"


def evaluate(eval_data: dict) -> dict:
    rows: list[dict] = []
    expected_tiers: list[str] = []
    predicted_tiers: list[str] = []
    baseline_tiers: list[str] = []

    for case in eval_data["evals"]:
        expected = case["expected_tier"]
        predicted = classify_prompt(case["prompt"])
        baseline = "deep"

        expected_tiers.append(expected)
        predicted_tiers.append(predicted)
        baseline_tiers.append(baseline)

        rows.append(
            {
                "id": case["id"],
                "prompt": case["prompt"],
                "expected_tier": expected,
                "predicted_tier": predicted,
                "baseline_tier": baseline,
                "correct": predicted == expected,
            }
        )

    total = len(rows)
    correct = sum(row["correct"] for row in rows)
    baseline_correct = sum(expected == "deep" for expected in expected_tiers)
    unsafe_downgrades = sum(
        row["expected_tier"] == "deep" and row["predicted_tier"] != "deep"
        for row in rows
    )
    overprovisioned = sum(
        TIER_TO_INDEX[row["predicted_tier"]] > TIER_TO_INDEX[row["expected_tier"]]
        for row in rows
    )

    predicted_cost = sum(TIER_COST[tier] for tier in predicted_tiers)
    baseline_cost = sum(TIER_COST[tier] for tier in baseline_tiers)
    oracle_cost = sum(TIER_COST[tier] for tier in expected_tiers)

    confusion = {
        expected: {predicted: 0 for predicted in TIERS} for expected in TIERS
    }
    for expected, predicted in zip(expected_tiers, predicted_tiers):
        confusion[expected][predicted] += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "skill_name": eval_data.get("skill_name", "billing-optimization-plan"),
        "total_cases": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "baseline_accuracy": round(baseline_correct / total, 4) if total else 0.0,
        "unsafe_downgrade_rate": round(unsafe_downgrades / total, 4) if total else 0.0,
        "overprovision_rate": round(overprovisioned / total, 4) if total else 0.0,
        "predicted_cost_units": predicted_cost,
        "baseline_cost_units": baseline_cost,
        "oracle_cost_units": oracle_cost,
        "estimated_cost_savings_vs_baseline_pct": round(
            100 * (baseline_cost - predicted_cost) / baseline_cost, 2
        )
        if baseline_cost
        else 0.0,
        "cost_gap_vs_oracle_units": predicted_cost - oracle_cost,
        "expected_distribution": dict(Counter(expected_tiers)),
        "predicted_distribution": dict(Counter(predicted_tiers)),
        "confusion_matrix": confusion,
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark billing-optimization-plan skill routing decisions."
    )
    parser.add_argument(
        "--evals",
        type=Path,
        default=Path(".agents/skills/billing-optimization-plan/evals/evals.json"),
        help="Path to evals.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".agents/skills/billing-optimization-plan/benchmarks/latest.json"),
        help="Path to benchmark output JSON",
    )
    args = parser.parse_args()

    eval_data = json.loads(args.evals.read_text(encoding="utf-8"))
    benchmark = evaluate(eval_data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    print(json.dumps(benchmark, indent=2))


if __name__ == "__main__":
    main()
