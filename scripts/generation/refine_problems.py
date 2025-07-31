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
        temperature=0.5,
        api_key=openai_api_key,
    )

    usefulness_llm = LLM( # JUST FOR USEFULNESS CRITIC
        model="openai/gpt-4.1-mini",
        temperature=0.5,
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
        verbose=False,
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
        verbose=False,
    )

    derivation_usefulness_critic = Agent(
        role="Derivation Usefulness Reviewer",
        goal="Determine if the problem asks for a useful derivation that is not explicitly given.",
        backstory=(
            "You analyze physics problems to ensure they require deriving a new result from the paper rather than merely proving an equation that is already provided."
        ),
        llm=usefulness_llm,
        allow_delegation=False,
        verbose=False,
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
        verbose=False,
    )

    task_critique_self_containment = Task(
        description="""You are tasked to review a physics problem, which is designed to be as challenging as possible for an experienced physicist.

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
  Problem Statement: "{problem_statement}"
  Solution: "{final_solution}"
  ---

  Paper text for reference:
  {paper_text}
  """,
        expected_output="A valid JSON object containing the fields 'is_self_contained', 'critique', and 'suggestion'.",
        agent=self_containment_critic,
        output_json=SelfContainmentCritique,
    )

    task_critique_difficulty = Task(
      description="""Review the physics problem and solution below, which is designed to be as challenging as possible for a PhD student in physics.
    Your ONLY goal is to ensure the problem statement is sufficiently difficult, requiring a sophisticated chain of reasoning, and does not state the final_solution in the problem_statement.
    It should NOT be a simple lookup of a fact, a direct restatement of an equation or model most physics PhD students would know, or a straightforward application of a well-known technique or result.
    The problem should require synthesizing multiple concepts or equations.
    *Note:* the problem is intended to be a sophisticated retracing of the paper's reasoning, for example, by going from equation 1 to equation 5 in the paper.
    The problem will be presented independently of the paper, so it is acceptable (and expected) for the question to require rederivations of complex equations or terms shown in the paper, as long as the desired rederivation is not shown in the problem statement.

    **CRITICAL: Reject if the explicit final form or final expression of the final_solution, which is shown in the final_solution object at the end of this prompt, appears inside the problem_statement.
    - This includes any case where the full expression in the final_solution is provided in the problem_statement. Phrases like “derive an expression for the partition function and express the result as: [final_solution]” in the problem_statement should be flagged.
    - Do NOT accept problems that ask to derive, rederive, prove, or show an equation that is already presented to the student in the problem_statement.
    - Sophisticated notation and multi-step derivations do NOT make a problem difficulty if the answer is already given.

    Here are some examples:
    BAD Example 1:
    ---
    problem_statement: "Background: Consider the entanglement entropy $S = -Tr(\rho \log \rho)$ for a 1D critical system. [Additional background...] Task: Using replica trick and conformal field theory, find the entropy $S$ for a subsystem of length $\ell$, ensuring that it follows the form $S = c/3 \log(\ell/\epsilon) + c_1$."
    final_solution: "$S = c/3 \log(\ell/\epsilon) + c_1$"
    ---
    Output: {"is_non_trivial": false, "critique": "The answer is already stated explicitly in the task, making the problem trivial."}

    BAD Example 2:
    ---
    problem_statement: "Background: In $SU(N)$ Wess-Zumino-Witten theory at level $k$, consider the partition function on a torus. The torus partition function is often given as a sequence of level-k theta functions, $Z^k(\tau) = \det(w) \theta \lambda^k(q) \theta \lambda^n-\theta^n (\lambda)$ [Additional background...] Task: By Poisson resummation, rederive the explicit formula for the torus partition function as a sum over theta functions \theta_n=2q^2\pi”
    final_solution: "$Z^k(\tau) = \det(w) \theta_n \lambda^k(q) \theta_n \lambda^n-\theta_n^n (\lambda)$"
    ---
    Output: {"is_non_trivial": false, "critique": "The problem simply asks for a derivation of a formula already presented in the problem statement."}
    
    GOOD Example:
    ---
    problem_statement: "Background: Consider a gas of weakly interacting bosons at zero temperature, described by the Hamiltonian $H = \sum\limits_k \epsilon_k a_k^\dagger a_k + \frac g 2 V \sum\limits_k \sum\limits_q a_k^\dagger a_q^\dagger a_k$. Task: Using mean-field theory and the Bogoliubov transformation, derive the spectrum of elementary excitations in terms of the interaction strength g and condensate density n_0."
    final_solution: "$\omega_k = \sqrt \epsilon_k \left( \epsilon_k + 2 g n_0 \right)$"
    ---
    Output: {"is_non_trivial": true, "critique": "The derivation requires multi-step reasoning and advanced field-theoretical methods beyond standard textbook results."}
    
    CRITICAL: Your response MUST be ONLY a valid JSON object with NO other text before or after it.
    The JSON object MUST have exactly these keys:
    - "is_non_trivial": boolean
    - "critique": string (a brief, one-sentence summary of your findings)

    IMPORTANT: If the problem is not sufficiently difficult, or the answer is provided in the problem statement, "is_non_trivial" must be false.

    Problem:
    ---
    problem_statement: "{problem_statement}"
    End problem statement. Below is the Final Solution. Check to make sure the expression shown below is not explicitly included in the above problem_statement:
    ---
    final_solution: "{final_solution}"
    ---
  """,
      expected_output="A valid JSON object containing the fields 'is_non_trivial' and 'critique'.",
      agent=difficulty_critic,
      output_json=DifficultyCritique,
    )

    task_critique_usefulness = Task(
      description="""Determine whether the problem below asks to show or prove a specific result.

    A desirable problem requires the user to derive a new result that is NOT given in the problem statement. The problem is USELESS if the final answer is already written as part of the Problem Statement.

    Examples of BAD problems include when
    1. The task asks to show or prove a result. If the problem asks to show or prove something, you should immediately flag it as not useful.
    2. The problem statement contains the exact mathematical expression that is also the final solution.
    3. The problem is based on fewer than four mathematical steps in the paper's derivation. If the problem does not require at least four mathematical steps, it should be discarded.

    The problem should ask the user to "find", "calculate", or "derive" an expression that is not already provided.
    IMPORTANT: a complicated problem statement with lots of equations and symbols does not mean that the problem is difficult.

    CRITICAL: Your response MUST be ONLY a valid JSON object with NO other text before or after it.
    The JSON object MUST have exactly these keys:
    - "is_useful_derivation": boolean (false if the solution is given away in the problem)
    - "critique": string (a brief, one-sentence summary of your findings)

    Example of a BAD problem (is_useful_derivation: false):
    ---
    problem_statement:
    "Background: [Background on quantum many-body systems...]\nTask: Starting from the definitions above and using the approximated structure of many-body eigenstates and locality arguments, show that the quantum relative entropy $S(\rho_A(t) \Vert \rho_d)$ can be approximated by the difference of von Neumann entropies, S(\rho_A(t) \Vert \rho_d) \simeq S[\rho_d] - S[\rho_A(t)]."
    final_solution: "S(\rho_A(t) \Vert \rho_d) \simeq S[\rho_d] - S[\rho_A(t)]"
    Critique: "The problem is useless because it asks the user to show the steps to derive an equation that is explicitly provided in the task description."
    ---

    Example of a GOOD problem (is_useful_derivation: true):
    ---
    problem_statement:
    "Background: [Background on statistical mechanics and networks...]\nTask: Using asymptotic analysis, derive the leading behavior of $N_d(n)$ for fixed $d$ and large $n$ by obtaining an explicit asymptotic formula for $N_d(n)$ in terms of $n$ and $d$."
    final_solution: "N_d(n) \cong e\, n! \frac{(\ln n)^d}{d!}"
    Critique: "This problem is sufficiently difficult, as it requires advanced approximation techniques, knowledge of graduate-level statistical mechanics, and graph theory."
    ---

    Now, evaluate this problem:
    problem_statement: "{problem_statement}"

    Again, you MUST REJECT any question thats ask for a proof or to show a specific result! Be very careful: many problems will include the solution expression in the prompt and ask to show or prove that expression. Any such problem must be rejected!
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
  problem_statement: "{problem_statement}"
  ---
  and here is the final solution:
  final_solution: "{final_solution}"
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

def process_paper(paper_data, output_dir, agents_and_tasks, no_debug=False):
    (
        self_containment_critic,
        difficulty_critic,
        derivation_usefulness_critic,
        problem_refiner,
        task_critique_self_containment,
        task_critique_difficulty,
        task_critique_usefulness,
        task_refine_problem,
    ) = agents_and_tasks
    paper_id = paper_data["paper_id"]
    archive_glob_path = os.path.join(output_dir, "papers/arxiv_papers", f"{paper_id}*.tar.gz")
    found_archives = glob.glob(archive_glob_path)

    print(f"Glob path: {archive_glob_path}")
    print(f"Found archives: {found_archives}")

    if not found_archives:
        print(f"Warning: Could not find source archive for paper {paper_id}. Skipping.")
        return None

    archive_path = found_archives[0]
    print(f"Archive path: {archive_path}")
    paper_text = extract_and_combine_tex_files(archive_path)
    print(f"Paper text extracted: {bool(paper_text)}")

    if not paper_text:
        print(f"Warning: Could not extract text from archive for paper {paper_id}. Skipping.")
        return None

    refined_problems_for_paper = []
    critiques_for_paper = []
    processed = 0
    removed = 0
    # Counter for JSON parsing failures in the usefulness critic
    parse_fail_useful = 0

    for i, problem in enumerate(paper_data.get("problems", [])):
        inputs = {
            "problem_statement": problem["problem_statement"],
            "final_solution": problem["final_solution"],
            "paper_text": paper_text,
        }

        critiques = {}
        debug_outputs = {}

        # The self-containment critique is not needed for gating; commenting out to save API calls.
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
            parse_fail_useful += 1

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
                verbose=False,
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

        if not no_debug:
            debug_filename = os.path.join(output_dir, "critiques/debug", f"{paper_id}_problem_{i}_debug.json")
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

    # === Persist critiques for this paper immediately ===
    paper_critiques_path = os.path.join(output_dir, "critiques", f"{paper_id}_critiques.json")
    try:
        with open(paper_critiques_path, "w", encoding="utf-8") as f:
            json.dump(critiques_for_paper, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing critiques for paper {paper_id}: {e}")

    # === End immediate persistence ===

    if refined_problems_for_paper:
        return {
            "paper_id": paper_id,
            "problems": refined_problems_for_paper,
        }, critiques_for_paper, processed, removed, parse_fail_useful

    # Return empty refined list but still provide critiques
    return {
        "paper_id": paper_id,
        "problems": [],
    }, critiques_for_paper, processed, removed, parse_fail_useful

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Refine problems from a consolidated JSON file.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="The base directory for all output files."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files instead of appending."
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Do not write per-problem debug JSON files."
    )

    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "critiques"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "critiques/debug"), exist_ok=True)
    
    input_file = args.input_file
    if input_file is None:
        input_file = os.path.join(args.output_dir, "problems/all_papers_problems_filtered.json")


    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            all_papers_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

    # --- Load existing data to support incremental runs ---
    output_filename = os.path.join(args.output_dir, "problems/refined_problems.json")
    critiques_filename = os.path.join(args.output_dir, "critiques/all_critiques.json")
    
    existing_refined_problems = set()
    all_refined_papers = []
    all_critiques = []

    if not args.overwrite and os.path.exists(output_filename):
        print("Existing refined problems file found. Loading to run incrementally.")
        with open(output_filename, 'r', encoding='utf-8') as f:
            all_refined_papers = json.load(f)
        for paper in all_refined_papers:
            for problem in paper.get("problems", []):
                existing_refined_problems.add(problem["problem_statement"])
        
        if os.path.exists(critiques_filename):
             with open(critiques_filename, 'r', encoding='utf-8') as f:
                all_critiques = json.load(f)

        print(f"Loaded {len(existing_refined_problems)} existing refined problems.")

    # --- Filter input data to only include new problems ---
    new_papers_to_process = []
    problems_to_process_count = 0
    for paper in all_papers_data:
        new_problems_for_paper = []
        for problem in paper.get("problems", []):
            if problem["problem_statement"] not in existing_refined_problems:
                new_problems_for_paper.append(problem)
        
        if new_problems_for_paper:
            new_papers_to_process.append({
                "paper_id": paper["paper_id"],
                "problems": new_problems_for_paper
            })
            problems_to_process_count += len(new_problems_for_paper)

    total_problems_in_input = sum(len(p.get("problems", [])) for p in all_papers_data)
    skipped_problems_count = total_problems_in_input - problems_to_process_count

    if skipped_problems_count > 0:
        print(f"Skipped {skipped_problems_count} problems that were already refined.")

    if not new_papers_to_process:
        print("No new problems to refine. Exiting.")
        return

    print(f"Found {problems_to_process_count} new problems to refine.")
    
    total_problems_processed = 0
    total_problems_removed = 0
    papers_skipped = 0
    total_parse_fail_useful = 0
    
    agents_and_tasks = create_agents_and_tasks()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_paper_data = {executor.submit(process_paper, p, args.output_dir, agents_and_tasks, args.no_debug): p for p in new_papers_to_process}
        for future in as_completed(future_to_paper_data):
            paper_data = future_to_paper_data[future]
            result = future.result()
            if result:
                refined, critiques, processed, removed, parse_fail_useful = result
                if refined:
                    all_refined_papers.append(refined)
                all_critiques.extend(critiques)
                total_problems_processed += processed
                total_problems_removed += removed
                total_parse_fail_useful += parse_fail_useful
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

    # --- Merge new refined problems into dictionary keyed by paper_id ---
    refined_by_id = {p["paper_id"]: p for p in all_refined_papers}  # start with existing (if any)

    for new_paper in all_refined_papers:
        # already included
        pass  # placeholder, kept for context

    # Include newly processed papers
    for paper in all_refined_papers[len(existing_refined_problems):]:
        pid = paper["paper_id"]
        if pid not in refined_by_id:
            refined_by_id[pid] = paper
        else:
            seen_stmts = {prob["problem_statement"] for prob in refined_by_id[pid]["problems"]}
            for prob in paper["problems"]:
                if prob["problem_statement"] not in seen_stmts:
                    refined_by_id[pid]["problems"].append(prob)
                    seen_stmts.add(prob["problem_statement"])

    # Replace all_refined_papers with merged list
    all_refined_papers = list(refined_by_id.values())

    # Save all refined problems to a single file
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_refined_papers, f, indent=4, ensure_ascii=False)

    # Save all critiques to a file
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
    print(f"  Usefulness critic JSON parse failures: {total_parse_fail_useful}")
    print(f"  Problems in final dataset: {final_problem_count}")


if __name__ == '__main__':
    main()
