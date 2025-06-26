import json
import glob
import os
import re

RAW_OUTPUT_DIR = "output/raw_json_outputs"
FILTERED_OUTPUT_FILENAME = "output/all_papers_problems_filtered.json"

def consolidate_and_filter():
    """
    Consolidates raw JSON outputs and filters them based on quality rules.
    """
    # Part 1: Consolidate raw JSON files
    json_files = glob.glob(os.path.join(RAW_OUTPUT_DIR, "*.json"))
    if not json_files:
        print(f"No JSON files found in '{RAW_OUTPUT_DIR}' to process.")
        return

    all_papers_data = []
    for file_path in sorted(json_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            all_papers_data.append(data)
        except json.JSONDecodeError as e:
            print(f"Warning: Skipping file due to JSON error: {file_path} - {e}")
        except Exception as e:
            print(f"Warning: Skipping file due to unexpected error: {file_path} - {e}")

    if not all_papers_data:
        print("No valid paper data was consolidated. Halting.")
        return
        
    print(f"Consolidated {len(all_papers_data)} papers from raw outputs.")

    # Part 2: Filter the consolidated problems
    filtered_papers_data = []
    total_problems_processed = 0
    total_problems_filtered = 0
    filter_reasons = {
        "empty_solution": 0,
        "missing_background_or_task": 0,
        "solution_in_problem_statement": 0,
        "extraneous_latex_in_problem_statement": 0,
        "solution_contains_prose": 0,
    }

    for paper_data in all_papers_data:
        total_problems_processed += len(paper_data.get("problems", []))
        
        filtered_problems_for_paper = []
        for problem in paper_data.get("problems", []):
            problem_statement = problem.get("problem_statement", "").strip()
            final_solution = problem.get("final_solution", "").strip()
            
            is_valid = True
            reason = ""

            if not final_solution:
                is_valid = False
                reason = "empty_solution"
            elif "Background:" not in problem_statement or "Task:" not in problem_statement:
                is_valid = False
                reason = "missing_background_or_task"
            elif "Solution:" in problem_statement:
                is_valid = False
                reason = "solution_in_problem_statement"
            elif any(keyword in problem_statement for keyword in [
                "\\documentclass", "\\usepackage", "\\begin{document}", "\\end{document}",
                "\\subsection", "\\subsubsection", "\\begin{problem}", "\\end{problem}", "```latex"
            ]):
                is_valid = False
                reason = "extraneous_latex_in_problem_statement"
            elif (re.search(r'^#+\s', final_solution, re.MULTILINE) or
                  re.search(r'^\*\s', final_solution, re.MULTILINE) or
                  re.search(r'^\d+\.\s', final_solution, re.MULTILINE) or
                  len(final_solution.split()) > 100):
                is_valid = False
                reason = "solution_contains_prose"

            if is_valid:
                filtered_problems_for_paper.append(problem)
            else:
                total_problems_filtered += 1
                if reason:
                    filter_reasons[reason] += 1
        
        if filtered_problems_for_paper:
            paper_data["problems"] = filtered_problems_for_paper
            filtered_papers_data.append(paper_data)

    with open(FILTERED_OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(filtered_papers_data, f, indent=4, ensure_ascii=False)

    print("\n--- Filtering Complete ---")
    print(f"Total problems processed: {total_problems_processed}")
    print(f"Problems filtered out: {total_problems_filtered}")
    print(f"Problems remaining: {total_problems_processed - total_problems_filtered}")
    
    print("\nReasons for filtering:")
    for reason, count in filter_reasons.items():
        if count > 0:
            print(f"- {reason.replace('_', ' ').capitalize()}: {count}")
    print(f"\nFiltered data saved to '{FILTERED_OUTPUT_FILENAME}'.")

if __name__ == "__main__":
    consolidate_and_filter() 