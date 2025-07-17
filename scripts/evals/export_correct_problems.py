#!/usr/bin/env python3
"""Extract problems that a given model answered perfectly.

Usage:

    # Basic usage - paths are inferred from the model name
    python export_correct_problems.py --model o3-mini

    # Explicitly specify paths
    python export_correct_problems.py \
        --model o3-mini \
        --benchmark path/to/your/benchmark_results_o3-mini.json \
        --output path/to/your/o3_correct_problems.json

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
    parser.add_argument("--model", required=True, help="Model key to inspect (e.g., 'o3-mini').")
    parser.add_argument("--benchmark", help="Path to benchmark_results JSON file. If not provided, it will be inferred from the model name.")
    parser.add_argument("--output", help="Destination JSON file for extracted problems. If not provided, it will be inferred from the model name.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it exists, instead of appending.")

    args = parser.parse_args()
    
    # Infer paths if not provided
    if not args.benchmark:
        args.benchmark = f"output/results/{args.model}/benchmark_results_{args.model}.json"
    
    if not args.output:
        args.output = f"output/problems/{args.model}_correct_problems.json"

    try:
        results = load_benchmark(args.benchmark)
    except Exception as exc:
        print(f"[ERROR] Failed to load benchmark file: {exc}")
        return

    newly_correct = collect_correct_problems(results, args.model)

    existing_correct = []
    if not args.overwrite and os.path.exists(args.output):
        print(f"Output file {args.output} exists. Appending new results.")
        try:
            with open(args.output, "r", encoding="utf-8") as fp:
                existing_correct = json.load(fp)
            if not isinstance(existing_correct, list):
                print(f"[WARN] Existing file {args.output} has an unexpected format and will be overwritten.")
                existing_correct = []
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"[WARN] Could not read or parse {args.output}. It will be overwritten.")
            existing_correct = []

    # Use a dictionary to merge and deduplicate problems
    all_problems_by_paper: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)

    # Process existing problems first
    for paper in existing_correct:
        paper_id = paper.get("paper_id")
        if paper_id:
            for problem in paper.get("problems", []):
                if "problem_statement" in problem:
                    all_problems_by_paper[paper_id][problem["problem_statement"]] = problem

    # Process newly found problems, overwriting duplicates to ensure data is fresh
    for paper in newly_correct:
        paper_id = paper.get("paper_id")
        if paper_id:
            for problem in paper.get("problems", []):
                if "problem_statement" in problem:
                    all_problems_by_paper[paper_id][problem["problem_statement"]] = problem

    # Reconstruct the list in the correct schema
    final_problems_list = [
        {"paper_id": pid, "problems": list(probs.values())}
        for pid, probs in all_problems_by_paper.items()
    ]

    if not final_problems_list:
        print("[WARN] No perfectly solved problems found to write.")
        return

    write_output(final_problems_list, args.output)

    total = sum(len(p["problems"]) for p in final_problems_list)
    print(f"Wrote {len(final_problems_list)} papers, {total} total problems to {args.output}.")


if __name__ == "__main__":
    main() 