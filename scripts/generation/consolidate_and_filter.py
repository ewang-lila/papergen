import json
import glob
import os
import re
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

RAW_OUTPUT_DIR = "output/papers/initial_QA_pairs"
FILTERED_OUTPUT_FILENAME = "output/problems/all_papers_problems_filtered.json"

def consolidate_and_filter():
    """
    Consolidates raw JSON outputs and filters them based on quality rules.
    """
    def extract_task(statement):
        """Extracts the 'Task' section from a problem statement."""
        match = re.search(r'Task:(.*)', statement, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    # Part 1: Load data from the consolidated all_papers.json
    consolidated_file_path = os.path.join(RAW_OUTPUT_DIR, "all_papers.json")
    
    if not os.path.exists(consolidated_file_path):
        print(f"Error: Consolidated file '{consolidated_file_path}' not found.")
        return

    try:
        with open(consolidated_file_path, 'r', encoding='utf-8') as f:
            all_papers_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {consolidated_file_path}: {e}")
        return
    except Exception as e:
        print(f"Unexpected error reading {consolidated_file_path}: {e}")
        return

    if not all_papers_data:
        print("No paper data found in the consolidated file. Halting.")
        return
        
    print(f"Loaded {len(all_papers_data)} papers from '{consolidated_file_path}'.")

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
        "duplicate_problem": 0,
        "prove_statement_in_task": 0,
        "problem_statement_too_long": 0,
        "solution_too_long": 0,
    }

    for paper_data in all_papers_data:
        total_problems_processed += len(paper_data.get("problems", []))
        
        filtered_problems_for_paper = []
        for problem in paper_data.get("problems", []):
            problem_statement = problem.get("problem_statement", "").strip()
            final_solution = problem.get("final_solution", "").strip()
            task_section = extract_task(problem_statement)
            
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
            # Require at least one whitespace character before 'prove' to avoid matching terms like 'improve'.
            elif re.search(r"\sprove\s+that\b|\sprove\s+this\b", task_section, re.IGNORECASE):
                is_valid = False
                reason = "prove_statement_in_task"

            if is_valid:
                filtered_problems_for_paper.append(problem)
            else:
                total_problems_filtered += 1
                if reason:
                    filter_reasons[reason] += 1
        
        if filtered_problems_for_paper:
            paper_data["problems"] = filtered_problems_for_paper
            filtered_papers_data.append(paper_data)

    # Part 3: Deduplicate problems using embeddings
    print("\n--- Deduplication Step ---")

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not found in a .env file.")
        print("Create a .env file with OPENAI_API_KEY='your-key' to enable deduplication.")
        print("Skipping deduplication step.")
    else:
        try:
            client = OpenAI()

            all_problems = []
            for paper in filtered_papers_data:
                all_problems.extend(paper.get("problems", []))

            if len(all_problems) < 2:
                print("Not enough problems to run deduplication.")
            else:
                print(f"Running deduplication on {len(all_problems)} problems.")
                
                problem_texts = [
                    extract_task(p.get("problem_statement", "")) + "\n\n" + p.get("final_solution", "")
                    for p in all_problems
                ]

                # --- Batch processing for embeddings to avoid API limits ---
                batch_size = 512  # A reasonable batch size to stay under token limits
                all_embeddings = []
                print(f"Processing embeddings in batches of {batch_size}...")

                for i in range(0, len(problem_texts), batch_size):
                    batch = problem_texts[i:i + batch_size]
                    
                    num_batches = (len(problem_texts) + batch_size - 1) // batch_size
                    print(f"  - Processing batch {i//batch_size + 1}/{num_batches}...")

                    response = client.embeddings.create(
                        input=batch,
                        model="text-embedding-3-small"
                    )
                    all_embeddings.extend([item.embedding for item in response.data])
                
                embeddings = np.array(all_embeddings)
                # --- End of batch processing change ---
                
                similarity_matrix = cosine_similarity(embeddings)
                
                duplicate_indices = set()
                similarity_threshold = 0.8
                print(f"Using similarity threshold: {similarity_threshold}")
                num_duplicates_found = 0
                for i in range(len(similarity_matrix)):
                    if i in duplicate_indices:
                        continue
                    for j in range(i + 1, len(similarity_matrix)):
                        if j in duplicate_indices:
                            continue
                        if similarity_matrix[i][j] > similarity_threshold:
                            duplicate_indices.add(j)
                            num_duplicates_found += 1
                            
                            problem_i_task = extract_task(all_problems[i].get("problem_statement", ""))
                            problem_j_task = extract_task(all_problems[j].get("problem_statement", ""))
                            
                            print(f"\n--- Found Duplicate Pair {num_duplicates_found} (Similarity: {similarity_matrix[i][j]:.4f}) ---")
                            print(f"  Keeping problem (index {i}):")
                            print(f"    Task: {problem_i_task[:150].replace(chr(10), ' ')}...")
                            print(f"    Solution: {all_problems[i].get('final_solution','')[:150].replace(chr(10), ' ')}...")
                            print(f"  Removing problem (index {j}):")
                            print(f"    Task: {problem_j_task[:150].replace(chr(10), ' ')}...")
                            print(f"    Solution: {all_problems[j].get('final_solution','')[:150].replace(chr(10), ' ')}...")
                            print("----------------------------------------------------------------------")


                num_duplicates = len(duplicate_indices)
                if num_duplicates > 0:
                    print(f"\nFound and removed {num_duplicates} duplicate problems in total.")
                    total_problems_filtered += num_duplicates
                    filter_reasons["duplicate_problem"] = num_duplicates
                    
                    problems_to_remove_ids = {id(all_problems[i]) for i in duplicate_indices}
                    
                    deduplicated_papers = []
                    for paper_data in filtered_papers_data:
                        original_problems = paper_data.get("problems", [])
                        deduplicated_problems_for_paper = [
                            p for p in original_problems if id(p) not in problems_to_remove_ids
                        ]
                        
                        if deduplicated_problems_for_paper:
                            paper_data["problems"] = deduplicated_problems_for_paper
                            deduplicated_papers.append(paper_data)
                    
                    filtered_papers_data = deduplicated_papers
                else:
                    print("No duplicates found.")

        except Exception as e:
            print(f"An error occurred during deduplication: {e}")
            print("Skipping deduplication step.")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(FILTERED_OUTPUT_FILENAME), exist_ok=True)

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