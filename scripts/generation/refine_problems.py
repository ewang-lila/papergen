import os
import sys
import json
import re
import argparse
import tarfile
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from crewai import Crew, Agent, Task, LLM
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment

google_api_key = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

class SelfContainmentCritique(BaseModel):
    is_self_contained: bool
    critique: str
    suggestion: str

class DifficultyCritique(BaseModel):
    is_non_trivial: bool
    critique: str

class UsefulDerivationCritique(BaseModel):
    is_useful_derivation: bool
    critique: str

class RefinedProblem(BaseModel):
    problem_statement: str
    final_solution: str

def create_agents_and_tasks():
    refiner_llm = LLM(
        model="openai/gpt-4.1",
        api_key=openai_api_key,
    )

    critic_llm = LLM(
        model="openai/gpt-4.1-mini",
        temperature=0.0,
        api_key=openai_api_key,
    )

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
        verbose=True,
    )

    difficulty_critic = Agent(
        role="Problem Difficulty and Triviality Reviewer",
        goal="Assess if a physics problem is non-trivial and requires a genuine chain of reasoning.",
        backstory=(
            "You are an expert physicist designing extremely challenging problems for PhD qualifying exams. "
            "Determine whether the problem is sufficiently difficult and requires an advanced, multi-step chain of reasoning designed to challenge the most advanced physics PhD students."
        ),
        llm=critic_llm,
        allow_delegation=False,
        verbose=True,
    )

    derivation_usefulness_critic = Agent(
        role="Derivation Usefulness Reviewer",
        goal="Determine if the problem asks for a useful derivation that is not explicitly given.",
        backstory=(
            "You analyze physics problems to ensure they require deriving a new result from the paper rather than merely verifying an equation that is already provided."
        ),
        llm=critic_llm,
        allow_delegation=False,
        verbose=True,
    )

    problem_refiner = Agent(
        role="Physics Problem Editor and Refiner",
        goal="Incorporate feedback from critics to refine a physics problem.",
        backstory=(
            "You will be given a physics problem, the paper it's based on, and critiques. You must address the feedback to produce a revised "
            "problem that is self-contained and challenging."
        ),
        llm=refiner_llm,
        allow_delegation=False,
        verbose=True,
    )

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
        agent=self_containment_critic,
        output_json=SelfContainmentCritique,
    )

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
      agent=difficulty_critic,
      output_json=DifficultyCritique,
    )

    task_critique_usefulness = Task(
      description="""Evaluate whether the problem below asks for a useful derivation from the paper.

A 'useful derivation' requires the user to derive a new and unseen result that is NOT given in the problem statement. The problem is USELESS if the solution is already contained in the problem statement or only asks for a proof or verification of a specific result. The problem is acceptable only if it requires several complex steps to reach the desired result, and to provide a new result unseen in the problem statement.

Check for these patterns of BAD problems:
1. The task asks to show or prove a specific result that is already shown.
2. The problem statement contains the exact mathematical expression that is also the final solution.
3. The problem is based on fewer than four steps in the paper's derivation.

The problem should ask the user to "find", "calculate", or "derive" an expression for a quantity that is not already provided AND requires significant extra work to derive.
IMPORTANT: a complicated problem statement with lots of equations and symbols does not mean that the problem is difficult.

Problem Statement: {problem_statement}
Final Solution: {final_solution}

CRITICAL: Your response MUST be ONLY a valid JSON object with NO other text before or after it.
The JSON object MUST have exactly these keys:
- "is_useful_derivation": boolean (false if the solution is given away in the problem)
- "critique": string (a brief, one-sentence summary of your findings)

Example of a BAD problem (is_useful_derivation: false):
---
Problem Statement:
"Background: [Background on quantum many-body systems...]\nTask: Starting from the definitions above and using the approximated structure of many-body eigenstates and locality arguments, derive that the quantum relative entropy $S(\rho_A(t) \Vert \rho_d)$ can be approximated by the difference of von Neumann entropies,
S(\rho_A(t) \Vert \rho_d) \simeq S[\rho_d] - S[\rho_A(t)]."
Final Solution: "S(\rho_A(t) \Vert \rho_d) \simeq S[\rho_d] - S[\rho_A(t)]"
Critique: "The problem is useless because it asks the user to derive an equation that is explicitly provided in the task description, making it trivial for the problem-solver since they already know the final solution."
---

Example of a GOOD problem (is_useful_derivation: true):
---
Problem Statement:
"Background: [Background on statistical mechanics and networks...]\nTask: Using asymptotic analysis, derive the leading behavior of $N_d(n)$ for fixed $d$ and large $n$, i.e., obtain an explicit asymptotic formula for $N_d(n)$ in terms of $n$ and $d$."
Final Solution: "N_d(n) \cong e\, n! \frac{(\ln n)^d}{d!}"
Critique: "This problem is sufficiently difficult, as it requires advanced approximation techniques, knowledge of graduate-level statistical mechanics, and graph theory."
---

Now, evaluate this problem:
Problem Statement: {problem_statement}
Final Solution: {final_solution}
""",
      expected_output="A valid JSON object containing the fields 'is_useful_derivation' and 'critique'.",
      agent=derivation_usefulness_critic,
      output_json=UsefulDerivationCritique,
    )

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
    "problem_statement": "Background: [problem text here]\n\nTask: [task description here]",
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
      context=[task_critique_self_containment, task_critique_difficulty, task_critique_usefulness],
      output_json=RefinedProblem,
    )

    return (
        self_containment_critic,
        difficulty_critic,
        derivation_usefulness_critic,
        problem_refiner,
        task_critique_self_containment,
        task_critique_difficulty,
        task_critique_usefulness,
        task_refine_problem,
    )


def extract_and_combine_tex_files(archive_path):
    """
    Extracts all .tex files from an arXiv source archive and combines them.
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

def process_paper(paper_data):
    (
        self_containment_critic,
        difficulty_critic,
        derivation_usefulness_critic,
        problem_refiner,
        task_critique_self_containment,
        task_critique_difficulty,
        task_critique_usefulness,
        task_refine_problem,
    ) = create_agents_and_tasks()
    paper_id = paper_data["paper_id"]
    archive_glob_path = os.path.join("output/papers/arxiv_papers", f"{paper_id}*.tar.gz")
    found_archives = glob.glob(archive_glob_path)

    if not found_archives:
        print(f"Warning: Could not find source archive for paper {paper_id}. Skipping.")
        return None

    archive_path = found_archives[0]
    paper_text = extract_and_combine_tex_files(archive_path)

    if not paper_text:
        print(f"Warning: Could not extract text from archive for paper {paper_id}. Skipping.")
        return None

    refined_problems_for_paper = []
    critiques_for_paper = []
    processed = 0
    removed = 0

    for i, problem in enumerate(paper_data.get("problems", [])):
        inputs = {
            "problem_statement": problem["problem_statement"],
            "final_solution": problem["final_solution"],
            "paper_text": paper_text,
        }

        critiques = {}
        debug_outputs = {}

        try:
            sc_crew = Crew(agents=[self_containment_critic], tasks=[task_critique_self_containment], verbose=False)
            sc_result = sc_crew.kickoff(inputs=inputs)
            sc_parsed = sc_result.json_dict
            if sc_parsed:
                critiques["self_containment"] = sc_parsed
            else:
                raise ValueError("Failed to get structured output from self-containment critique")
        except Exception as e:
            print(f"Error in self-containment critique: {e}")
            critiques["self_containment"] = {"error": str(e)}

        try:
            diff_crew = Crew(agents=[difficulty_critic], tasks=[task_critique_difficulty], verbose=False)
            diff_result = diff_crew.kickoff(inputs=inputs)
            diff_parsed = diff_result.json_dict
            if diff_parsed:
                critiques["difficulty"] = diff_parsed
            else:
                raise ValueError("Failed to get structured output from difficulty critique")
        except Exception as e:
            print(f"Error in difficulty critique: {e}")
            critiques["difficulty"] = {"error": str(e)}

        try:
            useful_crew = Crew(agents=[derivation_usefulness_critic], tasks=[task_critique_usefulness], verbose=False)
            useful_result = useful_crew.kickoff(inputs=inputs)
            useful_parsed = useful_result.json_dict
            if useful_parsed:
                critiques["useful_derivation"] = useful_parsed
            else:
                raise ValueError("Failed to get structured output from derivation usefulness critique")
        except Exception as e:
            print(f"Error in derivation usefulness critique: {e}")
            critiques["useful_derivation"] = {"error": str(e)}

        is_non_trivial = critiques.get("difficulty", {}).get("is_non_trivial", True)
        is_useful = critiques.get("useful_derivation", {}).get("is_useful_derivation", True)

        if not is_non_trivial:
            critique_text = critiques.get("difficulty", {}).get("critique", "No critique provided.")
            removed += 1
            critique_entry = {
                "paper_id": paper_id,
                "problem_index": i,
                "original_problem": {
                    "problem_statement": problem["problem_statement"],
                    "final_solution": problem["final_solution"],
                },
                "critiques": critiques,
                "removed": True,
                "removal_reason": f"Trivial problem: {critique_text}",
            }
            critiques_for_paper.append(critique_entry)
            processed += 1
            continue

        if not is_useful:
            critique_text = critiques.get("useful_derivation", {}).get("critique", "Marked as useless derivation")
            removed += 1
            critique_entry = {
                "paper_id": paper_id,
                "problem_index": i,
                "original_problem": {
                    "problem_statement": problem["problem_statement"],
                    "final_solution": problem["final_solution"],
                },
                "critiques": critiques,
                "removed": True,
                "removal_reason": f"Useless derivation: {critique_text}",
            }
            critiques_for_paper.append(critique_entry)
            processed += 1
            continue

        try:
            refiner_crew = Crew(
                agents=[self_containment_critic, difficulty_critic, derivation_usefulness_critic, problem_refiner],
                tasks=[task_critique_self_containment, task_critique_difficulty, task_critique_usefulness, task_refine_problem],
                verbose=True,
            )
            refiner_result = refiner_crew.kickoff(inputs=inputs)
            refined_parsed = refiner_result.json_dict
            if refined_parsed and "problem_statement" in refined_parsed and "final_solution" in refined_parsed:
                clean_refined_problem = {
                    "problem_statement": refined_parsed["problem_statement"],
                    "final_solution": refined_parsed["final_solution"],
                }
                refined_problems_for_paper.append(clean_refined_problem)

                critique_entry = {
                    "paper_id": paper_id,
                    "problem_index": i,
                    "original_problem": {
                        "problem_statement": problem["problem_statement"],
                        "final_solution": problem["final_solution"],
                    },
                    "critiques": critiques,
                    "refined_problem": clean_refined_problem,
                    "included_in_dataset": True,
                }
                critiques_for_paper.append(critique_entry)
            else:
                raise ValueError("Failed to get structured output or missing required fields from refinement")
        except Exception as e:
            was_non_trivial = critiques.get("difficulty", {}).get("is_non_trivial", False)
            was_useful = critiques.get("useful_derivation", {}).get("is_useful_derivation", False)
            if was_non_trivial and was_useful:
                clean_original = {
                    "problem_statement": problem["problem_statement"],
                    "final_solution": problem["final_solution"],
                }
                refined_problems_for_paper.append(clean_original)
                included = True
            else:
                removed += 1
                included = False

            critique_entry = {
                "paper_id": paper_id,
                "problem_index": i,
                "original_problem": {
                    "problem_statement": problem["problem_statement"],
                    "final_solution": problem["final_solution"],
                },
                "critiques": critiques,
                "refinement_error": str(e),
                "included_in_dataset": included,
            }
            critiques_for_paper.append(critique_entry)

        debug_filename = f"output/critiques/debug/{paper_id}_problem_{i}_debug.json"
        with open(debug_filename, "w", encoding="utf-8") as f:
            json.dump({
                "paper_id": paper_id,
                "problem_index": i,
                "problem_statement": problem["problem_statement"],
                "final_solution": problem["final_solution"],
                "debug_outputs": debug_outputs,
                "critiques_parsed": critiques,
            }, f, indent=2, ensure_ascii=False)

        processed += 1

    if refined_problems_for_paper:
        return {
            "paper_id": paper_id,
            "problems": refined_problems_for_paper,
        }, critiques_for_paper, processed, removed

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
        "--limit",
        type=int,
        default=None,
        help="Maximum number of problems to process (for testing). If not specified, all problems will be processed."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers to process papers"
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
    papers_skipped = 0
    total_problems_in_input = sum(len(p.get("problems", [])) for p in all_papers_data)
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_paper_data = {executor.submit(process_paper, p): p for p in all_papers_data}
        for future in as_completed(future_to_paper_data):
            paper_data = future_to_paper_data[future]
            result = future.result()
            if result:
                refined, critiques, processed, removed = result
                if refined:
                    all_refined_papers.append(refined)
                all_critiques.extend(critiques)
                total_problems_processed += processed
                total_problems_removed += removed
            else:
                # This indicates the paper was skipped, e.g., source not found.
                papers_skipped += 1
                print(f"Warning: Paper {paper_data.get('paper_id', 'Unknown')} was skipped and its problems were not processed.")

    seen_statements = set()
    deduped_papers = []
    for paper in all_refined_papers:
        unique_probs = []
        for prob in paper["problems"]:
            stmt = prob.get("problem_statement")
            if stmt not in seen_statements:
                seen_statements.add(stmt)
                unique_probs.append(prob)
        if unique_probs:
            deduped_papers.append({"paper_id": paper["paper_id"], "problems": unique_probs})
    all_refined_papers = deduped_papers

    if args.limit is not None:
        truncated = []
        count = 0
        for paper in all_refined_papers:
            new_probs = []
            for prob in paper["problems"]:
                if count >= args.limit:
                    break
                new_probs.append(prob)
                count += 1
            if new_probs:
                truncated.append({"paper_id": paper["paper_id"], "problems": new_probs})
            if count >= args.limit:
                break
        all_refined_papers = truncated

    # Save all refined problems to a single file
    output_filename = "output/problems/refined_problems.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_refined_papers, f, indent=4, ensure_ascii=False)

    # Save all critiques to a file
    critiques_filename = "output/critiques/all_critiques.json"
    with open(critiques_filename, 'w', encoding='utf-8') as f:
        json.dump(all_critiques, f, indent=4, ensure_ascii=False)

    print(f"\nAll papers processed. Refined problems saved to {output_filename}")
    print(f"Critiques saved to {critiques_filename}")
    
    final_problem_count = sum(len(p.get("problems", [])) for p in all_refined_papers)

    print(f"\n--- Refining Statistics ---")
    print(f"  Total problems in input file: {total_problems_in_input}")
    if papers_skipped > 0:
        print(f"  Papers skipped (source file not found): {papers_skipped}")
    print(f"  Total problems processed: {total_problems_processed}")
    print(f"  Problems removed (trivial/useless): {total_problems_removed}")
    print(f"  Problems in final dataset: {final_problem_count}")


if __name__ == '__main__':
    main()
