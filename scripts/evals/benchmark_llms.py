import openai
import anthropic
import requests
import urllib3
from google import genai
from google.genai import types
import json
import os
import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
import time

load_dotenv()

# Configure APIs
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENWEBUI_API_KEY = os.getenv("OPENWEBUI_API_KEY")
OPENWEBUI_API_BASE_URL = os.getenv("OPENWEBUI_API_BASE_URL", "https://open-webui.ml.lilasci.io/api")


SUPPORTED_MODELS = [
    "gpt-4o",
    "o4",
    "o3",
    "o3-mini",
    "o4-mini",
    "claude-4",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite-preview-06-17",
    "owui/qwen3:4b",
    "owui/qwen3:32b",
    "owui/qwen2.5:7b",
]

JUDGE_MODEL = "gpt-4.1"
JUDGE_PROMPT_FILE = os.path.join(os.path.dirname(__file__), "judge_prompt.txt")

MODEL_PROMPT = """You are an expert in physics. You will be given a physics problem. 
There is only one correct final answer to the problem. 
Provide your final, simplified answer inside a LaTeX \\boxed{{}} environment. \\n\\n"""

def extract_boxed_content(full_string: str):
    """
    Extracts content from the first \\boxed{...} environment in a string,
    correctly handling nested braces.
    """
    try:
        # Find the starting position of `\boxed{`
        start_index = full_string.find('\\boxed{')
        if start_index == -1:
            return None

        # Start searching for the matching brace after `\boxed{`
        search_start = start_index + len('\\boxed{')
        brace_level = 1
        for i in range(search_start, len(full_string)):
            if full_string[i] == '{':
                brace_level += 1
            elif full_string[i] == '}':
                brace_level -= 1
            
            if brace_level == 0:
                # We found the matching brace
                return full_string[search_start:i].strip()
        
        # If we reach here, a matching brace was not found
        return None
    except Exception:
        return None

def load_problems(filename):
    """Loads problems from the specified JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def get_model_response(model_name, problem_statement):
    """
    Gets a response from the specified LLM.
    """
    try:
        if model_name in ["o3", "o4", "gpt-4o", "o4-mini", "o3-mini"]:
            params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": MODEL_PROMPT},
                    {"role": "user", "content": problem_statement}
                ]
            }
            
            response = openai.chat.completions.create(**params)
            return response.choices[0].message.content
        elif model_name.startswith("owui/"):
            if not OPENWEBUI_API_KEY:
                return "Error: OPENWEBUI_API_KEY environment variable not set."
            
            headers = {
                "Authorization": f"Bearer {OPENWEBUI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model_name.replace("owui/", ""),
                "messages": [
                    {"role": "system", "content": MODEL_PROMPT},
                    {"role": "user", "content": problem_statement}
                ],
                "stream": False,
            }

            retries = 3
            initial_delay = 5
            backoff_factor = 2

            for attempt in range(retries):
                try:
                    response = requests.post(
                        f"{OPENWEBUI_API_BASE_URL}/chat/completions",
                        headers=headers,
                        json=payload,
                        verify=False,
                        proxies={"http": None, "https": None},  # Explicitly disable proxies
                        timeout=180,  # 3-minute timeout
                    )
                    response.raise_for_status()
                    return response.json()["choices"][0]["message"]["content"]
                except requests.exceptions.RequestException as e:
                    if attempt < retries - 1:
                        delay = initial_delay * (backoff_factor**attempt)
                        print(
                            f"Request for model {model_name} failed with error: {e}. "
                            f"Retrying in {delay}s... (Attempt {attempt+1}/{retries})"
                        )
                        time.sleep(delay)
                    else:
                        print(
                            f"Final attempt failed for model {model_name}. Error: {e}"
                        )
                        return f"Error getting response after {retries} attempts: {e}"

        elif "claude" in model_name.lower():
            # Anthropic API call
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            response_text = ""
            with client.messages.stream(
                model="claude-opus-4-20250514",
                max_tokens=16000,
                thinking={"type": "enabled", "budget_tokens": 8000},
                system=MODEL_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": problem_statement,
                    }
                ],
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta" and event.delta.type == "text_delta":
                        response_text += event.delta.text
            return response_text
        elif "gemini" in model_name.lower():
            # Google API call
            client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=MODEL_PROMPT + problem_statement,
                config=types.GenerateContentConfig(
                    max_output_tokens=16000
                )
            )
            return response.text
    except Exception as e:
        return f"Error getting response: {e}"

def get_judge_evaluation(problem_statement, ground_truth_solution, model_generated_expression):
    """
    Uses an LLM as a judge to evaluate the model's expression against the ground truth.
    """
    with open(JUDGE_PROMPT_FILE, 'r') as f:
        judge_prompt_template = f.read()

    prompt = judge_prompt_template.format(
        # problem_statement=problem_statement,
        ground_truth_solution=ground_truth_solution,
        model_generated_solution=model_generated_expression,
    )
    
    try:
        response = openai.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert in physics."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        evaluation = response.choices[0].message.content
        
        # Extract score
        score = None
        match = re.search(r"Score:\s*([0-9\.]+)", evaluation)
        if match:
            try:
                score = float(match.group(1))
            except (ValueError, IndexError):
                score = None
            
        return evaluation, score

    except Exception as e:
        return f"Error during evaluation: {e}", None

def calculate_summary_statistics(results, models):
    """Calculates summary statistics from the benchmark results."""
    summary = {model: {'1_count': 0, '0.5_count': 0, '0_count': 0, 'null_count': 0, 'total': 0, 'scores': []} for model in models}

    for res in results:
        for model_name, output in res.get("model_outputs", {}).items():
            if model_name not in summary:
                continue
            
            score = output.get("score")
            summary[model_name]['total'] += 1
            
            if score == 1.0:
                summary[model_name]['1_count'] += 1
                summary[model_name]['scores'].append(score)
            elif score == 0.5:
                summary[model_name]['0.5_count'] += 1
                summary[model_name]['scores'].append(score)
            elif score == 0.0:
                summary[model_name]['0_count'] += 1
                summary[model_name]['scores'].append(score)
            else:
                summary[model_name]['null_count'] += 1

    for model_name, stats in summary.items():
        if stats['scores']:
            stats['average_score'] = sum(stats['scores']) / len(stats['scores'])
        else:
            stats['average_score'] = 0.0
        del stats['scores']

    return summary


def evaluate_problem(problem, models):
    problem_result = problem.copy()
    problem_result["model_outputs"] = {}

    for model_name in models:
        model_solution_full = get_model_response(model_name, problem["problem_statement"])

        extracted_solution = extract_boxed_content(model_solution_full)

        evaluation = ""
        score = 0.0
        final_solution_for_json = ""

        if extracted_solution is not None:
            final_solution_for_json = extracted_solution
            evaluation, score = get_judge_evaluation(
                problem["problem_statement"],
                problem["ground_truth_solution"],
                extracted_solution,
            )
        else:
            score = 0.0
            evaluation = "Evaluation Error: No \\boxed{} expression found in the model's output."
            final_solution_for_json = model_solution_full

        problem_result["model_outputs"][model_name] = {
            "solution": final_solution_for_json,
            "evaluation": evaluation,
            "score": score,
        }

    return problem_result

def main():
    parser = argparse.ArgumentParser(description="Benchmark LLMs on physics problems.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="The base directory for all output files."
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=SUPPORTED_MODELS,
        choices=SUPPORTED_MODELS,
        help="Which models to benchmark.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of problems to evaluate."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to the JSON file with problems to benchmark."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,  # Set default to None to allow for dynamic generation
        help="File to save the benchmark results. If not provided, it will be generated automatically."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers to evaluate problems",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file instead of running incrementally."
    )
    args = parser.parse_args()

    # Dynamically set input and output filenames if not provided
    if args.input_file is None:
        args.input_file = os.path.join(args.output_dir, "problems/refined_problems.json")

    if args.output_file is None:
        input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
        
        if len(args.model) == 1:
            model_name = args.model[0].replace("openai/", "").replace("/", "-")
            output_dir_path = os.path.join(args.output_dir, "results", model_name)
            os.makedirs(output_dir_path, exist_ok=True)
            # Include input file identifier in the output filename
            args.output_file = os.path.join(output_dir_path, f"benchmark_results_{model_name}.json")
        else:
            # For multiple models, create a generic filename in the root of results
            output_dir_path = os.path.join(args.output_dir, "results")
            os.makedirs(output_dir_path, exist_ok=True)
            args.output_file = os.path.join(output_dir_path, f"benchmark_results_from_{input_basename}.json")
    else:
        # If an output file is specified, ensure its directory exists
        output_dir_path = os.path.dirname(args.output_file)
        if output_dir_path:
            os.makedirs(output_dir_path, exist_ok=True)

    # --- Load existing results for incremental benchmark ---
    existing_results = {}
    if not args.overwrite and os.path.exists(args.output_file):
        print(f"Loading existing benchmark results from {args.output_file} to run incrementally.")
        with open(args.output_file, 'r') as f:
            benchmark_data = json.load(f)
            # Create a lookup for existing results: (paper_id, problem_statement) -> result
            for res in benchmark_data.get("results", []):
                lookup_key = (res["paper_id"], res["problem_statement"])
                existing_results[lookup_key] = res

    # --- Prepare list of problems to evaluate ---
    papers = load_problems(args.input_file)
    all_problems = []
    for paper in papers:
        for problem in paper['problems']:
            all_problems.append({
                "paper_id": paper["paper_id"],
                "problem_statement": problem["problem_statement"],
                "ground_truth_solution": problem["final_solution"]
            })

    # --- Filter out problems that have already been evaluated for all specified models ---
    problems_to_evaluate = []
    final_results = list(existing_results.values()) # Start with existing results
    skipped_count = 0

    for problem in all_problems:
        lookup_key = (problem["paper_id"], problem["problem_statement"])
        if lookup_key in existing_results:
            # Check if all current models have been run for this problem
            existing_models = set(existing_results[lookup_key].get("model_outputs", {}).keys())
            if set(args.model).issubset(existing_models):
                skipped_count += 1
                continue  # Skip if all models are already present
        
        problems_to_evaluate.append(problem)
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} problems that were already evaluated for the specified models.")

    if not problems_to_evaluate:
        print("No new problems to evaluate for the specified models. Exiting.")
        return

    if args.limit:
        problems_to_evaluate = problems_to_evaluate[:args.limit]

    print(f"Found {len(problems_to_evaluate)} new problems to evaluate.")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(evaluate_problem, p, args.model) for p in problems_to_evaluate]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating problems"):
            new_result = fut.result()
            # Merge new results with existing ones if necessary
            lookup_key = (new_result["paper_id"], new_result["problem_statement"])
            if lookup_key in existing_results:
                existing_results[lookup_key]["model_outputs"].update(new_result["model_outputs"])
            else:
                final_results.append(new_result)

    # For any updated entries, we need to refresh them in the final list
    # This is a bit complex, might be easier to just rebuild the final list
    final_results_dict = {(r['paper_id'], r['problem_statement']): r for r in final_results}
    for res in problems_to_evaluate:
         lookup_key = (res["paper_id"], res["problem_statement"])
         if lookup_key in final_results_dict: # Should always be true
            # This part is complex. Let's simplify: just append and deduplicate later.
            pass # The logic in the ThreadPoolExecutor already handles updating or appending.

    summary = calculate_summary_statistics(final_results, args.model)

    final_output = {
        "results": final_results,
        "summary": summary
    }

    with open(args.output_file, 'w') as f:
        json.dump(final_output, f, indent=4)

    print(f"Benchmarking complete. Results and summary saved to {args.output_file}")


if __name__ == "__main__":
    main() 