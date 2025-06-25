import json
import os

INPUT_FILENAME = "output/all_papers_problems.json"
OUTPUT_FILENAME = "output/all_papers_problems_filtered.json"

def filter_problems():
    """
    Filters problems from the consolidated JSON file based on quality criteria.
    """
    if not os.path.exists(INPUT_FILENAME):
        print(f"Input file '{INPUT_FILENAME}' not found. Please run consolidate_output.py first.")
        return

    with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
        all_papers_data = json.load(f)

    filtered_papers_data = []
    total_problems_processed = 0
    total_problems_filtered = 0
    filter_reasons = {
        "empty_solution": 0,
        "missing_background_or_task": 0,
        "solution_in_problem_statement": 0,
        "solution_contains_prose": 0,
        "other_issues": 0,
    }

    for paper_data in all_papers_data:
        total_problems_processed += len(paper_data.get("problems", []))
        
        filtered_problems_for_paper = []
        for problem in paper_data.get("problems", []):
            problem_statement = problem.get("problem_statement", "").strip()
            final_solution = problem.get("final_solution", "").strip()
            
            is_valid = True
            reason = ""

            # Rule 1: Empty Solution
            if not final_solution:
                is_valid = False
                reason = "empty_solution"
            
            # Rule 2: Missing Background or Task
            elif "Background:" not in problem_statement or "Task:" not in problem_statement:
                is_valid = False
                reason = "missing_background_or_task"
            
            # Rule 3: Solution content accidentally in problem statement
            elif "Solution:" in problem_statement:
                is_valid = False
                reason = "solution_in_problem_statement"
            
            # Rule 4: Extraneous LaTeX boilerplate in problem statement
            elif any(keyword in problem_statement for keyword in [
                "\\documentclass", "\\usepackage", "\\begin{document}", "\\end{document}"
            ]):
                is_valid = False
                reason = "extraneous_latex_in_problem_statement"
            
            # Rule 5: Solution contains prose/markdown (heuristic for broken LaTeX output/parsing)
            # Check for common markdown headers or list items that should not be in a solution box
            elif (re.search(r'^#+\s', final_solution, re.MULTILINE) or # markdown headers
                  re.search(r'^\*\s', final_solution, re.MULTILINE) or # unordered lists
                  re.search(r'^\d+\.\s', final_solution, re.MULTILINE) or # ordered lists
                  len(final_solution.split()) > 100): # Arbitrary length check for prose
                is_valid = False
                reason = "solution_contains_prose"

            # Add other filtering rules as needed

            if is_valid:
                filtered_problems_for_paper.append(problem)
            else:
                total_problems_filtered += 1
                filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
        
        # Only add the paper if it still has problems after filtering
        if filtered_problems_for_paper:
            paper_data["problems"] = filtered_problems_for_paper
            filtered_papers_data.append(paper_data)

    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(filtered_papers_data, f, indent=4, ensure_ascii=False)

    print(f"Filtering complete. Original problems: {total_problems_processed}, Filtered out: {total_problems_filtered}")
    for reason, count in filter_reasons.items():
        if count > 0:
            print(f"- {reason.replace('_', ' ').capitalize()}: {count}")
    print(f"Filtered data saved to '{OUTPUT_FILENAME}'.")

if __name__ == "__main__":
    # Need to import re for regex in filter_problems
    import re
    filter_problems() 