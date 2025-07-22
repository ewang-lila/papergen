import os
import re
import json
import glob
import argparse
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from openai import OpenAI, APIStatusError
from pydantic import BaseModel, ValidationError

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

ARXIV_DOWNLOAD_DIR = "output/papers/arxiv_papers"

MAX_RETRIES = 3  # Number of attempts per problem if validation fails

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def extract_and_combine_tex_files(archive_path: str) -> Optional[str]:
    """Extract all .tex files from the arXiv source archive and concatenate them.

    This duplicates the logic used in *refine_problems.py* so we can run standalone
    without importing that heavy module (which pulls in crewai etc.).
    """
    extract_dir = os.path.join(
        os.path.dirname(archive_path),
        os.path.basename(archive_path).replace(".tar.gz", "_extracted"),
    )
    os.makedirs(extract_dir, exist_ok=True)

    combined = []

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tex_members = [m for m in tar.getmembers() if m.name.endswith(".tex")]
            for member in tex_members:
                tar.extract(member, path=extract_dir)

            # Heuristically prioritise a file that contains "main.tex" in its path.
            main_member = next((m for m in tex_members if "main.tex" in m.name.lower()), None)
            if main_member:
                tex_members.remove(main_member)
                tex_members.insert(0, main_member)

            for member in tex_members:
                fpath = os.path.join(extract_dir, member.name)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
                        combined.append(fp.read())
                except Exception:
                    # Skip unreadable files.
                    continue

        if combined:
            return "\n\n".join(combined)
    except tarfile.ReadError:
        # Not a valid tar.gz file.
        pass
    except Exception as e:
        print(f"Error extracting {archive_path}: {e}")

    return None


def normalise_latex(expr: str) -> str:
    """Normalise LaTeX expression for comparison."""
    # Remove common LaTeX wrappers and commands
    expr = re.sub(r'\\(bold|mathbf|math.*?)\{([^}]*)\}', r'\2', expr)  # Strip \bold{...}, \mathbf{...}, etc.
    expr = re.sub(r'\\(intercal|transp)', 'T', expr)  # Normalise transpose notations
    # Remove all whitespace
    expr = re.sub(r'\s+', '', expr)
    # To lowercase for case-insensitivity (if needed, but LaTeX is case-sensitive for commands, but variables might vary)
    # expr = expr.lower()  # Comment out if not desired
    return expr


# -----------------------------------------------------------------------------
# Prompt construction & parsing
# -----------------------------------------------------------------------------

def build_prompt(problem_statement: str, boxed_answer: str, paper_text: str) -> List[Dict[str, str]]:
    """Construct chat messages for the LLM call.

    We instruct the model to output a JSON with two keys so that parsing is easy.
    """
    system_msg = (
        "You are an expert physicist. "
        "Given a very challenging graduate-level physics problem, the final answer, "
        "and the full manuscript from which the problem and solution were derived, you must write a "
        "complete, rigorous, and precise step-by-step derivation that leads unambiguously to the same "
        "boxed result. Note that the problem and solution are completely derived from the paper; no additional information, assumptions, or context is required outside of the paper text.\n\n"  # Newlines for readability
        "OUTPUT FORMAT (CRITICAL): You MUST return ONLY a valid JSON object with exactly "
        "these two keys and no extra keys: \n"
        "  1. \"solution_trace\" - the full step-by-step solution explaining all the work required to reach the final result. Include the final \\boxed result at the end of your step-by-step solution. You MUST NOT reference any material from the paper (e.g., equation (3), reference (2), etc.).\n"
        "  2. \"final_solution\" - the boxed answer, identical to the provided one, e.g. \\boxed{E=mc^2}.\n"
        "Do not wrap the JSON in markdown fences, and do NOT include any additional commentary. DO NOT MAKE ANY ASSUMPTIONS OR GUESSES, AND DO NOT INTRODUCE ANY OUTSIDE INFORMATION."  # Important for parsing
    )

    user_content = (
        f"Problem Statement:\n{problem_statement}\n\n"
        f"Provided Final Answer:\n\\boxed{{{boxed_answer}}}\n\n"
        "Full Paper LaTeX (for reference):\n" + paper_text
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]


# -----------------------------------------------------------------------------
# Pydantic schema for LLM output
# -----------------------------------------------------------------------------

class LLMResponse(BaseModel):
    solution_trace: str
    final_solution: str

    class Config:
        extra = "ignore"  # Ignore any additional keys the model may return


def parse_llm_output(raw_text: str) -> Optional[Dict[str, Any]]:
    """Parse and validate the model output using Pydantic.

    The model *should* emit a JSON object, but sometimes it adds extra commentary or
    wraps the JSON in markdown fences. We try a few strategies to extract the JSON
    substring and then validate it against `LLMResponse`.
    """
    json_obj: Optional[Dict[str, Any]] = None

    # 1. Fast path – whole message is (hopefully) JSON.
    try:
        json_obj = json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # 2. Look inside markdown/code fences or anywhere in the text for a JSON blob.
    if json_obj is None:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if match:
            try:
                json_obj = json.loads(match.group(1))
            except json.JSONDecodeError:
                json_obj = None

    # 3. Fallback – first curly-brace pair in the text (greedy until last brace).
    if json_obj is None:
        brace_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if brace_match:
            try:
                json_obj = json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                json_obj = None

    # Validate with Pydantic.
    if isinstance(json_obj, dict):
        try:
            validated = LLMResponse.model_validate(json_obj)
            return validated.model_dump()
        except ValidationError:
            return None

    return None


# -----------------------------------------------------------------------------
# Core processing
# -----------------------------------------------------------------------------

def process_problem(openai_client: OpenAI, paper_text: str, problem: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Generate and validate a solution trace for a single problem."""
    problem_statement = problem.get("problem_statement", "")
    original_boxed = problem.get("final_solution", "")

    if not problem_statement or not original_boxed:
        print("[WARN] Missing statement or final solution; skipping problem.")
        return None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            messages = build_prompt(problem_statement, original_boxed, paper_text)
            response = openai_client.chat.completions.create(
                model="gpt-4.1",  # or your preferred model; must support JSON mode
                messages=messages,
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content.strip()
            parsed = parse_llm_output(raw)
            if not parsed or "solution_trace" not in parsed or "final_solution" not in parsed:
                print(f"[Attempt {attempt}] Failed to parse JSON output. Retrying…")
                continue

            trace = parsed["solution_trace"].strip()
            final_sol = parsed["final_solution"].strip()

            # Accept the model output as-is without further validation.
            return {
                "problem_statement": problem_statement,
                "solution_trace": trace,
                "final_solution": final_sol,
            }
        except APIStatusError as api_err:
            print(f"OpenAI API error (status {api_err.status_code}): {api_err}")
            break  # Abort further retries on persistent errors
        except Exception as e:
            print(f"Unexpected error during LLM call: {e}")
            continue

    print("Exceeded maximum retries – skipping problem.")
    return None


def process_paper(openai_client: OpenAI, paper_entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process all problems for a single paper and return enriched data."""
    paper_id = paper_entry.get("paper_id")
    if not paper_id:
        print("[ERROR] Paper entry missing paper_id – skipping.")
        return None

    print(f"[{paper_id}] Looking for paper archive...")
    archive_pattern = os.path.join(ARXIV_DOWNLOAD_DIR, f"{paper_id}*.tar.gz")
    archives = glob.glob(archive_pattern)
    if not archives:
        print(f"[WARN] No archive found for paper {paper_id}. Skipping problems from this paper.")
        return None

    paper_text = extract_and_combine_tex_files(archives[0])
    if not paper_text:
        print(f"[WARN] Could not extract LaTeX for {paper_id}. Skipping.")
        return None

    enriched_problems = []
    problems_to_process = paper_entry.get("problems", [])
    
    print(f"[{paper_id}] Found {len(problems_to_process)} problems. Starting processing...")
    for i, prob in enumerate(problems_to_process):
        print(f"[{paper_id}] Processing problem {i+1}/{len(problems_to_process)}...")
        enriched = process_problem(openai_client, paper_text, prob)
        if enriched:
            enriched_problems.append(enriched)

    if enriched_problems:
        return {"paper_id": paper_id, "problems": enriched_problems}
    return None

# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------

def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not found. Set it in your environment or .env file.")
        return

    parser = argparse.ArgumentParser(description="Generate step-by-step solution traces for refined physics problems.")
    parser.add_argument("--model", required=True, help="Model name to use for generating traces (e.g., 'o3-mini').")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel threads (OpenAI calls).")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of problems to process from the input file."
    )
    args = parser.parse_args()

    # Construct file paths based on model name
    input_file = f"output/problems/{args.model}_correct_problems.json"
    output_file = f"output/problems/{args.model}_solution_traces.json"

    # Load refined problems.
    try:
        with open(input_file, "r", encoding="utf-8") as fp:
            all_papers_data: List[Dict[str, Any]] = json.load(fp)
    except Exception as e:
        print(f"[ERROR] Failed to read input problems file: {e}")
        return

    # Apply problem limit if specified
    papers_to_process = all_papers_data
    if args.limit and args.limit > 0:
        print(f"Limiting processing to the first {args.limit} problems.")
        
        limited_papers = []
        problems_count = 0
        
        for paper in all_papers_data:
            if problems_count >= args.limit:
                break

            problems_in_paper = paper.get("problems", [])
            if not problems_in_paper:
                continue

            num_to_take = args.limit - problems_count
            
            if len(problems_in_paper) <= num_to_take:
                # Take all problems from this paper
                limited_papers.append(paper)
                problems_count += len(problems_in_paper)
            else:
                # Take a subset of problems from this paper
                limited_paper_copy = paper.copy()
                limited_paper_copy["problems"] = problems_in_paper[:num_to_take]
                limited_papers.append(limited_paper_copy)
                problems_count += num_to_take
        
        papers_to_process = limited_papers

    openai_client = OpenAI()

    results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_pid = {
            executor.submit(process_paper, openai_client, paper): paper.get("paper_id")
            for paper in papers_to_process
        }
        for fut in as_completed(future_to_pid):
            pid = future_to_pid[fut]
            try:
                paper_result = fut.result()
                if paper_result:
                    results.append(paper_result)
                    print(f"[DONE] Processed paper {pid} – {len(paper_result['problems'])} traces.")
                else:
                    print(f"[SKIP] No traces generated for {pid}.")
            except Exception as e:
                print(f"[ERROR] Exception while processing paper {pid}: {e}")

    # Ensure output directory exists.
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as out_fp:
        json.dump(results, out_fp, indent=4, ensure_ascii=False)

    total_traces = sum(len(p["problems"]) for p in results)
    print(f"\nGenerated {total_traces} solution traces across {len(results)} papers. Saved to {output_file}.")


if __name__ == "__main__":
    main() 