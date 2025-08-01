"""Combine problems from two filtered-problem files, reuse any existing solution traces,
then generate missing traces with OpenAI.

Usage example:

    python scripts/generation/combine_and_generate_traces.py \
        --original-dir output \
        --new-dir output/new_problems \
        --output-dir output/all_problems \
        --workers 8

This will:
1. Read  <original-dir>/problems/all_papers_problems_filtered.json  and
   <new-dir>/problems/all_papers_problems_filtered.json.
2. Merge them (deduplicating by `problem_statement`).
3. Search *both* source directories for files matching  *_solution_traces.json  and
   reuse any traces already generated.
4. Generate missing traces (in parallel) and save a fully-traced dataset to
   <output-dir>/problems/solution_traces_all.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

from dotenv import load_dotenv

# Re-use the heavy-lifting trace generator. The project isn't a Python package,
# so we add the repository root to sys.path before importing.
import pathlib, sys
_repo_root = pathlib.Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.append(str(_repo_root))

from scripts.generation.generate_solution_traces import (
    process_paper,  # type: ignore
    OpenAI,         # imported transitively
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def load_filtered(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"[WARN] Filtered-problems file not found: {path}")
        return []
    obj = read_json(path)
    if not isinstance(obj, list):
        print(f"[ERROR] Expected list in {path}")
        return []
    return obj


def build_trace_lookup(trace_files: List[str]) -> Dict[str, Dict[str, Any]]:
    """Return mapping  problem_statement -> enriched_problem_dict."""
    lookup: Dict[str, Dict[str, Any]] = {}
    for tf in trace_files:
        try:
            data = read_json(tf)
        except Exception as exc:
            print(f"[WARN] Failed to load {tf}: {exc}")
            continue
        for paper in data:
            for prob in paper.get("problems", []):
                stmt = prob.get("problem_statement")
                if stmt:
                    lookup[stmt] = prob
    return lookup


def merge_problem_lists(lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Merge papers, deduplicate by problem_statement."""
    by_paper: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    for papers in lists:
        for paper in papers:
            pid = paper.get("paper_id")
            if not pid:
                continue
            for prob in paper.get("problems", []):
                stmt = prob.get("problem_statement")
                if stmt:
                    by_paper[pid][stmt] = prob
    # Convert back to list[paper]
    merged: List[Dict[str, Any]] = []
    for pid, probs in by_paper.items():
        merged.append({"paper_id": pid, "problems": list(probs.values())})
    return merged


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser(description="Combine problems and generate missing solution traces.")
    ap.add_argument("--original-dir", default="output", help="Original base output directory.")
    ap.add_argument("--new-dir", default="output/new_problems", help="Second output directory to merge.")
    ap.add_argument("--output-dir", default="output/all_problems", help="Where to write the combined/traced dataset.")
    ap.add_argument("--workers", type=int, default=8, help="Parallel OpenAI workers for trace generation.")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.output_dir, "problems"), exist_ok=True)

    # 1. Load filtered problem files
    orig_filtered = load_filtered(os.path.join(args.original_dir, "problems/all_papers_problems_filtered.json"))
    new_filtered = load_filtered(os.path.join(args.new_dir, "problems/all_papers_problems_filtered.json"))

    combined_papers = merge_problem_lists([orig_filtered, new_filtered])
    combined_path = os.path.join(args.output_dir, "problems/all_papers_problems_combined.json")
    with open(combined_path, "w", encoding="utf-8") as fp:
        json.dump(combined_papers, fp, indent=2, ensure_ascii=False)
    print(f"Wrote combined problems file with {sum(len(p['problems']) for p in combined_papers)} problems -> {combined_path}")

    # 2. Build lookup of existing traces
    trace_files = glob.glob(os.path.join(args.original_dir, "problems/*_solution_traces.json")) + \
                 glob.glob(os.path.join(args.new_dir, "problems/*_solution_traces.json"))
    trace_lookup = build_trace_lookup(trace_files)
    print(f"Loaded {len(trace_lookup)} existing solution traces from {len(trace_files)} files.")

    # 3. Attach existing traces & collect problems that still need them
    papers_to_generate: List[Dict[str, Any]] = []
    for paper in combined_papers:
        new_probs: List[Dict[str, Any]] = []
        for prob in paper.get("problems", []):
            stmt = prob.get("problem_statement")
            if stmt in trace_lookup:
                # Reuse existing enriched problem (which already contains solution_trace)
                prob.update(trace_lookup[stmt])
            else:
                new_probs.append(prob)
        if new_probs:
            papers_to_generate.append({"paper_id": paper["paper_id"], "problems": new_probs})

    print(f"Need to generate traces for {sum(len(p['problems']) for p in papers_to_generate)} problems.")
    if not papers_to_generate:
        final_out = os.path.join(args.output_dir, "problems/solution_traces_all.json")
        with open(final_out, "w", encoding="utf-8") as fp:
            json.dump(combined_papers, fp, indent=2, ensure_ascii=False)
        print(f"All traces already present. Dataset saved to {final_out}")
        return

    # 4. Generate missing traces (reusing helpers from generate_solution_traces)
    def _select_base_dir(pid: str) -> str:
        """Return original-dir or new-dir depending on where the archive exists."""
        pattern_orig = os.path.join(args.original_dir, "papers/arxiv_papers", f"{pid}*.tar.gz")
        if glob.glob(pattern_orig):
            return args.original_dir
        pattern_new = os.path.join(args.new_dir, "papers/arxiv_papers", f"{pid}*.tar.gz")
        if glob.glob(pattern_new):
            return args.new_dir
        # Fallback to original
        return args.original_dir

    client = OpenAI()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for paper in papers_to_generate:
            pid = paper["paper_id"]
            base_dir = _select_base_dir(pid)
            futs[ex.submit(process_paper, client, paper, base_dir)] = pid
        for fut in as_completed(futs):
            pid = futs[fut]
            res = fut.result()
            if res:
                # Integrate new traces back into combined_papers structure
                lookup_paper = {p["paper_id"]: p for p in combined_papers}
                dest = lookup_paper.get(pid)
                if not dest:
                    # Should not happen, but guard anyway
                    combined_papers.append(res)
                else:
                    for pr in res["problems"]:
                        stmt = pr["problem_statement"]
                        updated = False
                        for existing in dest["problems"]:
                            if existing.get("problem_statement") == stmt:
                                existing.update(pr)
                                updated = True
                                break
                        if not updated:
                            dest["problems"].append(pr)
            else:
                print(f"[WARN] Failed to generate traces for paper {pid}")

    # 5. Save final dataset
    final_out = os.path.join(args.output_dir, "problems/solution_traces_all.json")
    with open(final_out, "w", encoding="utf-8") as fp:
        json.dump(combined_papers, fp, indent=2, ensure_ascii=False)
    print(f"Finished. Dataset with traces saved to {final_out}")


if __name__ == "__main__":
    main()
