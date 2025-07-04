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
google_api_key = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- LLM Configuration ---

# Set up the LLM for the refiner agent, using the more powerful model
refiner_llm = LLM(
    model="openai/gpt-4.1",
    api_key=openai_api_key
)

# Set up a separate, faster LLM for the critic agents
critic_llm = LLM(
    model="openai/gpt-4.1-mini",
    temperature=0.0,
    api_key=openai_api_key, 
)

# --- Agents ---

# 1. Self-Containment Critic Agent
self_containment_critic = Agent(
    role="Problem Self-Containment Reviewer",
    goal="Review a physics problem to ensure it is self-contained.",
    backstory=(
        "You are an expert physicist specializing in designing challenging problems for graduate students. "
        "The problems are created using research papers. Make sure that there are no references to the paper in the problem statement "
        "or to variables that are not self-defined (unless they would be obvious to a professional physicist)."
    ),
    llm=critic_llm,
    allow_delegation=False,
    verbose=True
)

# 2. Difficulty Critic Agent
difficulty_critic = Agent(
    role="Problem Difficulty and Triviality Reviewer",
    goal="Assess if a physics problem is non-trivial and requires a genuine chain of reasoning.",
    backstory=(
        "You are an expert physicist designing extremely challenging problems for PhD qualifying exams. "
        "Determine whether the problem is sufficiently difficult and requires an advanced, multi-step chain of reasoning designed to challenge the most advanced physics PhD students."
    ),
    llm=critic_llm,
    allow_delegation=False,
    verbose=True
)

# 3. Problem Refiner Agent
problem_refiner = Agent(
    role="Physics Problem Editor and Refiner",
    goal="Incorporate feedback from critics to refine a physics problem.",
    backstory=(
        "You will be given a physics problem, the paper it's based on, and critiques. You must address the feedback to produce a revised "
        "problem that is self-contained and challenging."
    ),
    llm=refiner_llm,
    allow_delegation=False,
    verbose=True
)

# --- Tasks ---

# Task for the Self-Containment Critic
task_critique_self_containment = Task(
    description="""Review the physics problem below, which is designed to be as challenging as possible for an experienced physicist.
  
    Problem Statement: {problem_statement}
    Final Solution: {final_solution}
  
    Your ONLY goal is to ensure it is self-contained.
    Check for:
    1. Any references to the source paper (e.g., "as seen in the paper" or "as shown in Fig. 4") or outside references (e.g., "as shown in [2]").
    2. Any empirical results or data that are not explicitly provided in the problem statement.
    3. Any variables or terms that are used in the solution but not clearly introduced in the problem statement. It is okay to use variables that are introduced in the paper, but they must be clearly introduced in the problem statement.
    4. Note that this problem is designed to be as challenging as possible for someone with a PhD in physics, so you should not nit-pick or obsess over statements and variables that a physicist would be able to reasonably infer.
    
    CRITICAL: Your response MUST be ONLY a valid JSON object with NO other text before or after it.
    The JSON object MUST have exactly these keys:
    - "is_self_contained": boolean
    - "critique": string (a brief, one-sentence summary of your findings, empty string if is_self_contained is true)
    - "suggestion": string (one concrete fix for the issue, empty string if is_self_contained is true)

  If the problem is satisfactory, "is_self_contained" should be true and "critique" should be an empty list. Only provide suggestions if the problem is not self-contained; do not nitpick on minor details or make the problem easier if it is already self-contained.

  Example of correct output:
  {"is_self_contained": false, "critique": "The term $v_{\mu}$ is used in the final solution but not defined in the problem statement, despite being defined as the frictional velocity in the main text.", "suggestion": "Add the phrase, 'where $v_{\mu} is the frictional velocity' at the end of the problem statement."}

  Problem:
  ---
  Problem Statement: {problem_statement}
  Solution: {final_solution}
  ---

  Paper text for reference:
  {paper_text}
  """,
  expected_output="A valid JSON object containing the fields 'is_self_contained', 'critique', and 'suggestion'.",
  agent=self_containment_critic
)

# Task for the Difficulty Critic
task_critique_difficulty = Task(
  description="""Review the physics problem below, which is designed to be as challenging as possible for a PhD student in physics.
  Your ONLY goal is to ensure the problem is sufficiently difficult, requiring a sophisticated chain of reasoning.
  It should NOT be a simple lookup of a fact, a direct restatement of an equation or model most physics PhD students would know, or a straightforward application of a well-known technique or result.
  The problem should require synthesizing multiple concepts or equations. 
  *Note:* the problem is intended to be a sophisticated retracing of the paper's reasoning, for example, by going from equation 1 to equation 5 in the paper. 
  The problem will be presented independently of the paper, so it is acceptable (and expected) for the question to require rederivations of complex equations or terms shown in the paper.

  CRITICAL: Your response MUST be ONLY a valid JSON object with NO other text before or after it.
  The JSON object MUST have exactly these keys:
  - "is_non_trivial": boolean
  - "critique": string (a brief, one-sentence summary of your findings)
  
  IMPORTANT: If the problem is not sufficiently difficult, "is_non_trivial" must be false.
  
  Example of correct output:
  {"is_non_trivial": false, "critique": "The problem asks for an alternate form of a equation covered in most graduate quantum field theory courses."}
  {"is_non_trivial": true, "critique": "The question requires detailed knowledge of statistical mechanics and field theory, as well as the application of the Laplace transform to derive an approximate analytical solution."}
  
  Problem:
  ---
  Problem Statement: {problem_statement}
  Final Solution: {final_solution}
  ---

  """,
  expected_output="A valid JSON object containing the fields 'is_non_trivial' and 'critique'.",
  agent=difficulty_critic
)

# Task for the Refiner Agent
task_refine_problem = Task(
  description="""Your task is to refine a physics problem based on specific feedback from two expert critics.
  You must address every issue raised in the critiques you are provided, but only address the issues that are raised. If no issues are raised, you must leave the problem as is.
  Use the original problem and answer as a reference.
  
  CRITICAL: Your output MUST be a valid JSON object with EXACTLY these two keys:
  - "problem_statement": The full refined problem statement, following the "Background:", "Task:", and "Solution:" template.
  - "final_solution": The unchanged final solution from the original problem.
  
  DO NOT include any text before or after the JSON object.
  DO NOT include any markdown, especially code blocks like```json.
  DO NOT include any other keys or metadata.
  ONLY USE proper LaTeX formatting. Under no circumstances should you use any special characters or unicode characters in your response; use *only* LaTeX commands for ALL symbols and characters. If there are unicode characters in the original problem statement, you must replace them with LaTeX commands.

  Example of correct output format:
  {
    "problem_statement": "Background: [problem text here]\\n\\nTask: [task description here]",
    "final_solution": "[solution expression here]"
  }

  Now, here is the original problem:
  ---
  problem_statement: {problem_statement}
  final_solution: {final_solution}
  ---

  Review the critiques that follow and revise the problem if necessary. Do not make any changes if no issues are raised. Make sure all LaTeX is wrapped in $$. Do not use any special characters or unicode characters in your response; replace any of these characters in the original problem-solution with proper LaTeX!
  """,
  expected_output="A valid JSON object with exactly two keys: 'problem_statement' and 'final_solution'. No other text or formatting.",
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

def parse_json_output(output_str):
    """
    Extracts the first valid JSON object from a string that may contain other text.
    Handles JSON in markdown code blocks as well.
    """
    try:
        # 1. Look for JSON in markdown code blocks first. This is often a reliable indicator.
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\s*```', output_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Found markdown block but failed to parse JSON: {e}")
                # Continue, as there might be another JSON object outside the block

        # 2. If no valid JSON in markdown, find the first substring that is a valid JSON object.
        # This is more robust against preceding text (like "Thought: ...").
        for match in re.finditer(r'{', output_str):
            start_index = match.start()
            substring = output_str[start_index:]
            try:
                # Use a decoder to find the first valid JSON object in the substring
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(substring)
                return obj  # Return the first valid object found
            except json.JSONDecodeError:
                # This '{' was not the start of a valid JSON object, so we continue.
                continue
        
        # 3. If no JSON object is found, return None.
        print("Warning: Could not find any valid JSON object in the output string.")
        return None

    except Exception as e:
        print(f"Error parsing JSON output: {e}")
        return None

# --- Main Execution ---

def main():
    os.makedirs("output/critiques", exist_ok=True)
    os.makedirs("output/critiques/debug", exist_ok=True)
    
    parser = argparse.ArgumentParser(description="Refine problems from a consolidated JSON file.")
    parser.add_argument(
        "--input-file",
        type=str,
        default="output/problems/all_papers_problems_filtered.json",
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
    total_problems_removed = 0

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

            # Execute critic tasks individually
            critiques = {}
            debug_outputs = {}
            
            # Self-containment critique
            try:
                sc_crew = Crew(
                    agents=[self_containment_critic],
                    tasks=[task_critique_self_containment],
                    verbose=False
                )
                sc_result = sc_crew.kickoff(inputs=inputs)
                sc_str = str(sc_result.raw) if hasattr(sc_result, 'raw') else str(sc_result)
                
                debug_outputs["self_containment_raw"] = sc_str
                
                # Parse JSON output
                sc_parsed = parse_json_output(sc_str)
                if sc_parsed:
                    critiques["self_containment"] = sc_parsed
                else:
                    raise ValueError("Failed to parse JSON output")
                
            except Exception as e:
                print(f"Error in self-containment critique: {e}")
                critiques["self_containment"] = {"error": str(e)}
            
            # Difficulty critique
            try:
                diff_crew = Crew(
                    agents=[difficulty_critic],
                    tasks=[task_critique_difficulty],
                    verbose=False
                )
                diff_result = diff_crew.kickoff(inputs=inputs)
                diff_str = str(diff_result.raw) if hasattr(diff_result, 'raw') else str(diff_result)
                
                debug_outputs["difficulty_raw"] = diff_str
                
                # Parse JSON output
                diff_parsed = parse_json_output(diff_str)
                if diff_parsed:
                    critiques["difficulty"] = diff_parsed
                else:
                    raise ValueError("Failed to parse JSON output")
                
            except Exception as e:
                print(f"Error in difficulty critique: {e}")
                critiques["difficulty"] = {"error": str(e)}
            
            # Check if the problem is trivial and should be removed
            is_non_trivial = critiques.get("difficulty", {}).get("is_non_trivial", True)  # Default to non-trivial (don't remove)

            if not is_non_trivial:
                critique_text = critiques.get("difficulty", {}).get("critique", "No critique provided.")
                print(f"Problem marked for removal: problem is trivial. Critique: {critique_text}")
                total_problems_removed += 1

                critique_entry = {
                    "paper_id": paper_id,
                    "problem_index": i,
                    "original_problem": {
                        "problem_statement": problem["problem_statement"],
                        "final_solution": problem["final_solution"]
                    },
                    "critiques": critiques,
                    "removed": True,
                    "removal_reason": f"Trivial problem: {critique_text}"
                }
                all_critiques.append(critique_entry)
                total_problems_processed += 1
                continue

            # Now run the refiner
            try:
                refiner_crew = Crew(
                    agents=[self_containment_critic, difficulty_critic, problem_refiner],
                    tasks=[task_critique_self_containment, task_critique_difficulty, task_refine_problem],
                    verbose=True
                )
                refiner_result = refiner_crew.kickoff(inputs=inputs)
                
                # Get the raw output
                refiner_raw = str(refiner_result.raw) if hasattr(refiner_result, 'raw') else str(refiner_result)
                debug_outputs["refiner_raw"] = refiner_raw
                
                # Parse JSON output
                refined_parsed = parse_json_output(refiner_raw)
                if refined_parsed and "problem_statement" in refined_parsed and "final_solution" in refined_parsed:
                    clean_refined_problem = {
                        "problem_statement": str(refined_parsed["problem_statement"]),
                        "final_solution": str(refined_parsed["final_solution"])
                    }
                    refined_problems_for_paper.append(clean_refined_problem)
                    
                    critique_entry = {
                        "paper_id": paper_id,
                        "problem_index": i,
                        "original_problem": {
                            "problem_statement": problem["problem_statement"],
                            "final_solution": problem["final_solution"]
                        },
                        "critiques": critiques,
                        "refined_problem": clean_refined_problem,
                        "included_in_dataset": True
                    }
                    all_critiques.append(critique_entry)
                else:
                    raise ValueError("Failed to parse JSON output or missing required fields")
                    
            except Exception as e:
                print(f"Warning: Failed to refine problem: {e}")
                
                # Include original if it was non-trivial
                was_non_trivial = critiques.get("difficulty", {}).get("is_non_trivial", False)
                if was_non_trivial:
                    print(f"Including original non-trivial problem despite refinement error")
                    clean_original = {
                        "problem_statement": problem["problem_statement"],
                        "final_solution": problem["final_solution"]
                    }
                    refined_problems_for_paper.append(clean_original)
                    included = True
                else:
                    print(f"Excluding trivial problem that failed refinement")
                    total_problems_removed += 1
                    included = False
                
                critique_entry = {
                    "paper_id": paper_id,
                    "problem_index": i,
                    "original_problem": {
                        "problem_statement": problem["problem_statement"],
                        "final_solution": problem["final_solution"]
                    },
                    "critiques": critiques,
                    "refinement_error": str(e),
                    "included_in_dataset": included
                }
                all_critiques.append(critique_entry)
            
            # Save debug outputs
            debug_filename = f"output/critiques/debug/{paper_id}_problem_{i}_debug.json"
            with open(debug_filename, 'w', encoding='utf-8') as f:
                json.dump({
                    "paper_id": paper_id,
                    "problem_index": i,
                    "problem_statement": problem["problem_statement"],
                    "final_solution": problem["final_solution"],
                    "debug_outputs": debug_outputs,
                    "critiques_parsed": critiques
                }, f, indent=2, ensure_ascii=False)
            
            total_problems_processed += 1
        
        if refined_problems_for_paper:
            all_refined_papers.append({
                "paper_id": paper_id,
                "problems": refined_problems_for_paper
            })

    # Save all refined problems to a single file
    output_filename = "output/problems/revised_problems.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_refined_papers, f, indent=4, ensure_ascii=False)

    # Save all critiques to a file
    critiques_filename = "output/critiques/all_critiques.json"
    with open(critiques_filename, 'w', encoding='utf-8') as f:
        json.dump(all_critiques, f, indent=4, ensure_ascii=False)

    print(f"\nAll papers processed. Refined problems saved to {output_filename}")
    print(f"Critiques saved to {critiques_filename}")
    print(f"\nStatistics:")
    print(f"  Total problems processed: {total_problems_processed}")
    print(f"  Problems removed (too trivial): {total_problems_removed}")
    print(f"  Problems in final dataset: {total_problems_processed - total_problems_removed}")


if __name__ == '__main__':
    main() 