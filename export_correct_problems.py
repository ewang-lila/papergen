#!/usr/bin/env python3
"""Extract problems that a given model answered perfectly.

Usage (defaults shown):

    python export_correct_problems.py \
        --benchmark output/results/o3-mini/benchmark_results_o3-mini.json \
        --model o3-mini \
        --output output/problems/o3_correct_problems.json

The script reads the benchmark results file (format produced by *benchmark_llms.py*),
filters for problems where the chosen model has score == 1 (or 1.0), and
writes a new JSON file in the same schema as *refined_problems.json* so that it
can be fed directly into *generate_solution_traces.py*.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List


DEFAULT_BENCHMARK_PATH = "output/results/o3-mini/benchmark_results_o3-mini.json"
DEFAULT_OUTPUT_PATH = "output/problems/o3_correct_problems.json"


def load_benchmark(path: str) -> List[Dict[str, Any]]:
    """Load benchmark JSON and return list of result entries."""
    with open(path, "r", encoding="utf-8") as fp:
        obj = json.load(fp)

    # Some benchmark files store the list under the key "results",
    # others may already be a list.
    if isinstance(obj, dict) and "results" in obj:
        return obj["results"]
    if isinstance(obj, list):
        return obj

    raise ValueError("Unrecognized benchmark JSON structure – expected list or dict with 'results'.")


def collect_correct_problems(results: List[Dict[str, Any]], model_key: str) -> List[Dict[str, Any]]:
    """Return problems where model_key scored 1."""
    papers: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for entry in results:
        paper_id = entry.get("paper_id")
        if not paper_id:
            continue

        # Extract score
        score = (
            entry.get("model_outputs", {})
            .get(model_key, {})
            .get("score")
        )
        # Accept scores that equal 1 within floating error
        if score is None or abs(float(score) - 1.0) > 1e-6:
            continue

        papers[paper_id].append(
            {
                "problem_statement": entry.get("problem_statement", ""),
                "final_solution": entry.get("ground_truth_solution", ""),
            }
        )

    # Convert to list in required schema
    return [
        {"paper_id": pid, "problems": probs} for pid, probs in papers.items()
    ]


def write_output(correct: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(correct, fp, indent=4, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export problems a model solved perfectly to a new JSON file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK_PATH, help="Path to benchmark_results JSON file.")
    parser.add_argument("--model", default="o3-mini", help="Model key to inspect (e.g., 'o3-mini').")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Destination JSON file for extracted problems.")

    args = parser.parse_args()

    try:
        results = load_benchmark(args.benchmark)
    except Exception as exc:
        print(f"[ERROR] Failed to load benchmark file: {exc}")
        return

    correct = collect_correct_problems(results, args.model)

    if not correct:
        print("[WARN] No perfectly solved problems found for the specified model.")
        return

    write_output(correct, args.output)

    total = sum(len(p["problems"]) for p in correct)
    print(f"Wrote {len(correct)} papers, {total} total problems to {args.output}.")


if __name__ == "__main__":
    main() 