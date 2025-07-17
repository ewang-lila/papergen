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
from tqdm import tqdm
from dotenv import load_dotenv

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
    "owui/qwen3:32b"
]

JUDGE_MODEL = "gpt-4.1-mini"
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

            response = requests.post(
                f"{OPENWEBUI_API_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                verify=False,
                proxies={ "http": None, "https": None } # Explicitly disable proxies
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
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

def main():
    parser = argparse.ArgumentParser(description="Benchmark LLMs on physics problems.")
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
        default="output/problems/refined_problems.json",
        help="Path to the JSON file with problems to benchmark."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,  # Set default to None to allow for dynamic generation
        help="File to save the benchmark results. If not provided, it will be generated automatically."
    )
    args = parser.parse_args()

    # Dynamically set output filename if not provided
    if args.output_file is None:
        input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
        
        if len(args.model) == 1:
            model_name = args.model[0].replace("openai/", "").replace("/", "-")
            output_dir = f"output/results/{model_name}"
            os.makedirs(output_dir, exist_ok=True)
            # Include input file identifier in the output filename
            args.output_file = f"{output_dir}/benchmark_results_{model_name}_from_{input_basename}.json"
        else:
            # For multiple models, create a generic filename in the root of results
            output_dir = "output/results"
            os.makedirs(output_dir, exist_ok=True)
            args.output_file = f"{output_dir}/benchmark_results_from_{input_basename}.json"
    else:
        # If an output file is specified, ensure its directory exists
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    papers = load_problems(args.input_file)
    results = []

    all_problems = []
    for paper in papers:
        for problem in paper['problems']:
            all_problems.append({
                "paper_id": paper["paper_id"],
                "problem_statement": problem["problem_statement"],
                "ground_truth_solution": problem["final_solution"]
            })

    if args.limit:
        all_problems = all_problems[:args.limit]

    for problem in tqdm(all_problems, desc="Evaluating problems"):
        problem_result = problem.copy()
        problem_result["model_outputs"] = {}

        for model_name in tqdm(args.model, desc=f"Benchmarking models on {problem['paper_id']}", leave=False):
            model_solution_full = get_model_response(model_name, problem["problem_statement"])
            
            # Extract the boxed part of the model's solution using the robust function
            extracted_solution = extract_boxed_content(model_solution_full)
            
            evaluation = ""
            score = 0.0
            final_solution_for_json = ""

            if extracted_solution is not None:
                final_solution_for_json = extracted_solution
                
                # Get evaluation only if a valid solution is found
                evaluation, score = get_judge_evaluation(
                    problem["problem_statement"],
                    problem["ground_truth_solution"],
                    extracted_solution
                )
            else:
                # If no boxed solution is found, the answer is wrong.
                score = 0.0
                evaluation = "Evaluation Error: No \\boxed{} expression found in the model's output."
                # Store the full response for debugging purposes
                final_solution_for_json = model_solution_full
            
            problem_result["model_outputs"][model_name] = {
                "solution": final_solution_for_json,
                "evaluation": evaluation,
                "score": score,
            }

        results.append(problem_result)

    summary = calculate_summary_statistics(results, args.model)

    final_output = {
        "results": results,
        "summary": summary
    }

    with open(args.output_file, 'w') as f:
        json.dump(final_output, f, indent=4)

    print(f"Benchmarking complete. Results and summary saved to {args.output_file}")


if __name__ == "__main__":
    main() 