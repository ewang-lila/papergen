import os
import sys
import json
import re
import argparse
import tarfile
import glob
from crewai import Crew, Agent, Task, LLM
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment
api_key = os.getenv("GOOGLE_API_KEY")

# --- LLM Configuration ---

# Set up the LLM for the refiner agent, using the more powerful model
refiner_llm = LLM(
    model="gemini/gemini-2.5-pro-preview-05-06",
    temperature=0.2,
    api_key=api_key
)

# Set up a separate, faster LLM for the critic agents
critic_llm = LLM(
    model="gemini/gemini-2.0-flash-lite",
    temperature=0.0,
    api_key=api_key
)

# --- Agents ---

# 1. Self-Containment Critic Agent
self_containment_critic = Agent(
    role="Problem Self-Containment Auditor",
    goal="Review a physics problem to ensure it is entirely self-contained. The problem must not reference the source paper or use undefined variables. Provide feedback in a structured JSON format.",
    backstory=(
        "You are a meticulous editor specializing in scientific and technical content. "
        "Your sole focus is identifying any information in a problem statement that is not explicitly provided. "
        "A perfect problem is a closed logical universe."
    ),
    llm=critic_llm,
    allow_delegation=False,
    verbose=True
)

# 2. Difficulty Critic Agent
difficulty_critic = Agent(
    role="Problem Difficulty and Triviality Analyst",
    goal="Assess if a physics problem is non-trivial and requires a genuine chain of reasoning. Provide feedback in a structured JSON format.",
    backstory=(
        "You are an expert in physics. "
        "Using the provided source paper, you will judge if the problem is non-trivial and requires a genuine, multi-step chain of reasoning designed to challenge advanced physics PhD students."
    ),
    llm=critic_llm,
    allow_delegation=False,
    verbose=True
)

# 4. Problem Refiner Agent
problem_refiner = Agent(
    role="Physics Problem Editor and Refiner",
    goal="Rewrite a physics problem to perfection by incorporating all feedback from two different critics (self-containment and difficulty).",
    backstory=(
        "You are an expert in physics. You may receive feedback from two different critics about a physics problem. If they give you feedback, you must incorporate their suggestions and create a revised problem."
        "If no feedback is given, do not change the problem."
    ),
    llm=refiner_llm,
    allow_delegation=False,
    verbose=True
)

# --- Tasks ---

# Task for the Self-Containment Critic
task_critique_self_containment = Task(
  description="""Review the physics problem below, which is designed to be as challenging as possible.
  Your ONLY goal is to ensure it is self-contained.
  Check for:
  1. Any references to the source paper (e.g., "as seen in the paper" or "as shown in Fig. 4") or outside references (e.g., "as shown in [2]").
  2. Any empirical results or data that are not explicitly provided in the problem statement.
  3. Any variables or terms that are used in the solution but not clearly introduced in the problem statement. It is okay to use variables that are introduced in the paper, but they must be clearly introduced in the problem statement. Similarly, it is okay to use variables that must be derived during the solution, but the problem statement must clearly state that the variable must be derived.

  Provide your feedback as a JSON object with the following keys:
  - "is_self_contained": boolean
  - "critique": string (a brief, one-sentence summary of your findings)
  - "issues": list of objects, where each object has:
    - "finding": string (the specific text that is problematic)
    - "suggestion": string (a concrete way to fix the issue)

  If the problem is perfect, "is_self_contained" should be true and "issues" should be an empty list. Only provide suggestions if the problem is not self-contained; do not nitpick on minor details or make the problem easier if it is already self-contained.

  Problem:
  ---
  Problem Statement: {problem_statement}
  Final Solution: {final_solution}
  ---
  """,
  expected_output="A valid JSON object containing the fields 'is_self_contained', 'critique', and 'issues'.",
  agent=self_containment_critic
)

# Task for the Difficulty Critic
task_critique_difficulty = Task(
  description="""Review the physics problem below, which is designed to be as challenging as possible.
  Your ONLY goal is to ensure the problem is non-trivial and requires a sophisticated chain of reasoning.
  It should NOT be a simple lookup of a fact or a direct restatement of an equation from the paper.
  Use the provided paper text to judge if the problem requires synthesizing multiple concepts or equations.

  Provide your feedback as a JSON object with the following keys:
  - "is_non_trivial": boolean
  - "critique": string (a brief, one-sentence summary of your findings)
  - "issues": list of objects, where each object has:
    - "finding": string (why the problem is trivial)
    - "suggestion": string (a concrete way to make the problem more challenging, e.g., 'require the user to combine equation X and Y')

  If the problem's difficulty is good, "is_non_trivial" should be true and "issues" should be an empty list. Only provide suggestions if the problem is currently trivial or does not target a difficult aspect of the paper.

  Problem:
  ---
  Problem Statement: {problem_statement}
  Final Solution: {final_solution}
  ---

  Full Paper Text for Context:
  ---
  {paper_text}
  ---
  """,
  expected_output="A valid JSON object containing the fields 'is_non_trivial', 'critique', and 'issues'.",
  agent=difficulty_critic
)

# Task for the Refiner Agent
task_refine_problem = Task(
  description="""Your task is to refine a physics problem based on specific feedback from two expert critics.
  You must address every issue raised in the critiques you are provided, but only address the issues that are raised.
  Use the original paper text as a reference to ensure scientific accuracy, as well as the original problem and answer..
  The final output MUST be a JSON object with two keys: "problem_statement" and "final_solution". The answer must be the same as the original answer. You must also follow the the "Background:", "Task:", and "Solution:" template.
  This is the template you must follow to provide the problem statement and solution:



  Original Problem:
  ---
  problem_statement: {problem_statement}
  final_solution: {final_solution}
  ---

  Full Paper Text for Context:
  ---
  {paper_text}
  ---

  Review the critiques that follow and rewrite the problem to perfection.
  """,
  expected_output="A single, valid JSON object with the keys 'Problem Statement and 'Final Solution. The 'Problem Statement should be the full, refined problem statement. The 'Final Solution should be the unchanged final solution from the original problem.",
  agent=problem_refiner,
  context=[task_critique_self_containment, task_critique_difficulty]
)

def extract_and_combine_tex_files(archive_path):
    """
    Extracts all .tex files from an arXiv source archive and combines them.
    (Copied from arxiv_processor.py)
    """
    extract_dir = os.path.join(os.path.dirname(archive_path), os.path.basename(archive_path).replace(".tar.gz", "_extracted"))
    os.makedirs(extract_dir, exist_ok=True)
    
    combined_content = ""
    
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tex_files_members = [m for m in tar.getmembers() if m.name.endswith(".tex")]
            
            # Extract all tex files
            for member in tex_files_members:
                tar.extract(member, path=extract_dir)

            main_tex_member = None
            for m in tex_files_members:
                if 'main.tex' in m.name.lower():
                    main_tex_member = m
                    break
            
            if main_tex_member:
                tex_files_members.remove(main_tex_member)
                tex_files_members.insert(0, main_tex_member)

            for member in tex_files_members:
                file_path = os.path.join(extract_dir, member.name)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    combined_content += f.read() + "\n\n"

        if combined_content:
            print(f"Successfully extracted and combined {len(tex_files_members)} .tex files from {archive_path}")
            return combined_content
        else:
            print(f"No .tex files found in {archive_path}")

    except tarfile.ReadError:
        print(f"Could not open {archive_path} as a tar.gz file.")
    except Exception as e:
        print(f"Error processing {archive_path}: {e}")
        
    return None

# --- Main Execution ---

def main():
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/critiques", exist_ok=True)
    parser = argparse.ArgumentParser(description="Refine problems from a consolidated JSON file.")
    parser.add_argument(
        "--input-file",
        type=str,
        default="output/all_papers_problems_filtered.json",
        help="Path to the consolidated JSON file with problems to refine."
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Maximum number of problems to process (for testing). If not specified, all problems will be processed."
    )
    args = parser.parse_args()

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            all_papers_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

    all_refined_papers = []
    all_critiques = []
    total_problems_processed = 0

    for paper_data in all_papers_data:
        # Check if we've reached the limit
        if args.max_problems and total_problems_processed >= args.max_problems:
            print(f"\nReached maximum number of problems to process ({args.max_problems}). Stopping.")
            break
            
        paper_id = paper_data["paper_id"]
        print(f"\n\n--- Processing Paper: {paper_id} ---")

        # Find the paper's source archive
        archive_glob_path = os.path.join("output/arxiv_papers", f"{paper_id}*.tar.gz")
        found_archives = glob.glob(archive_glob_path)
        
        if not found_archives:
            print(f"Warning: Could not find source archive for paper {paper_id}. Skipping.")
            continue
        
        archive_path = found_archives[0]
        paper_text = extract_and_combine_tex_files(archive_path)

        if not paper_text:
            print(f"Warning: Could not extract text from archive for paper {paper_id}. Skipping.")
            continue

        refined_problems_for_paper = []
        for i, problem in enumerate(paper_data.get("problems", [])):
            # Check if we've reached the limit
            if args.max_problems and total_problems_processed >= args.max_problems:
                print(f"\nReached maximum number of problems to process ({args.max_problems}). Stopping.")
                break
                
            print(f"\n--- Refining Problem {i+1}/{len(paper_data['problems'])} for paper {paper_id} ---")
            print(f"Total problems processed so far: {total_problems_processed}")

            inputs = {
                "problem_statement": problem["problem_statement"],
                "final_solution": problem["final_solution"],
                "paper_text": paper_text
            }

            # Execute critic tasks individually to capture their outputs
            critiques = {}
            
            # Self-containment critique
            try:
                sc_crew = Crew(
                    agents=[self_containment_critic],
                    tasks=[task_critique_self_containment],
                    verbose=False
                )
                sc_result = sc_crew.kickoff(inputs=inputs)
                sc_str = str(sc_result.raw) if hasattr(sc_result, 'raw') else str(sc_result)
                json_match = re.search(r'```json\n({.*?})\n```', sc_str, re.DOTALL)
                if json_match:
                    critiques["self_containment"] = json.loads(json_match.group(1))
                else:
                    critiques["self_containment"] = json.loads(sc_str)
            except Exception as e:
                print(f"Error in self-containment critique: {e}")
                critiques["self_containment"] = {"error": "Failed to parse", "raw": str(e)}
            
            # Difficulty critique
            try:
                diff_crew = Crew(
                    agents=[difficulty_critic],
                    tasks=[task_critique_difficulty],
                    verbose=False
                )
                diff_result = diff_crew.kickoff(inputs=inputs)
                diff_str = str(diff_result.raw) if hasattr(diff_result, 'raw') else str(diff_result)
                json_match = re.search(r'```json\n({.*?})\n```', diff_str, re.DOTALL)
                if json_match:
                    critiques["difficulty"] = json.loads(json_match.group(1))
                else:
                    critiques["difficulty"] = json.loads(diff_str)
            except Exception as e:
                print(f"Error in difficulty critique: {e}")
                critiques["difficulty"] = {"error": "Failed to parse", "raw": str(e)}
                
            # Now run the refiner with all the tasks
            refiner_crew = Crew(
                agents=[self_containment_critic, difficulty_critic, problem_refiner],
                tasks=[task_critique_self_containment, task_critique_difficulty, task_refine_problem],
                verbose=True
            )
            refiner_result = refiner_crew.kickoff(inputs=inputs)
            
            try:
                # Convert CrewOutput to string
                result_str = str(refiner_result.raw) if hasattr(refiner_result, 'raw') else str(refiner_result)
                
                # Try to extract JSON from markdown code block
                json_str_match = re.search(r'```json\s*\n(.*?)\n```', result_str, re.DOTALL)
                if json_str_match:
                    json_str = json_str_match.group(1)
                    # Fix common JSON issues with LaTeX
                    # Replace double backslashes with single for LaTeX commands
                    json_str = json_str.replace('\\\\', '\\')
                    refined_problem = json.loads(json_str)
                else:
                    # Try parsing the entire string as JSON
                    refined_problem = json.loads(result_str)
                
                # Ensure we have the expected keys
                if "problem_statement" not in refined_problem and "question" in refined_problem:
                    refined_problem["problem_statement"] = refined_problem["question"]
                if "final_solution" not in refined_problem and "answer" in refined_problem:
                    refined_problem["final_solution"] = refined_problem["answer"]
                    
                refined_problems_for_paper.append(refined_problem)
                    
                # Save critique data
                critique_entry = {
                    "paper_id": paper_id,
                    "problem_index": i,
                    "original_problem": {
                        "problem_statement": problem["problem_statement"],
                        "final_solution": problem["final_solution"]
                    },
                    "critiques": critiques,
                    "refined_problem": refined_problem
                }
                all_critiques.append(critique_entry)
                    
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"Warning: Could not parse final JSON output for problem. Storing raw output.")
                refined_problems_for_paper.append({"error": "failed to parse output", "raw_output": result_str})
                
                # Still save critique data even if refinement failed
                critique_entry = {
                    "paper_id": paper_id,
                    "problem_index": i,
                    "original_problem": {
                        "problem_statement": problem["problem_statement"],
                        "final_solution": problem["final_solution"]
                    },
                    "critiques": critiques,
                    "refined_problem": {"error": "failed to parse output", "raw_output": result_str}
                }
                all_critiques.append(critique_entry)
            
            total_problems_processed += 1
        
        if refined_problems_for_paper:
            all_refined_papers.append({
                "paper_id": paper_id,
                "problems": refined_problems_for_paper
            })

    # Save all refined problems to a single file
    output_filename = "output/revised_problems.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_refined_papers, f, indent=4, ensure_ascii=False)

    # Save all critiques to a file
    critiques_filename = "output/critiques/all_critiques.json"
    with open(critiques_filename, 'w', encoding='utf-8') as f:
        json.dump(all_critiques, f, indent=4, ensure_ascii=False)

    print(f"\nAll papers processed. Refined problems saved to {output_filename}")
    print(f"Critiques saved to {critiques_filename}")


if __name__ == '__main__':
    main() 