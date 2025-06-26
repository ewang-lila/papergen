import openai
import anthropic
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

SUPPORTED_MODELS = [
    "4o",
    "o4",
    "o3",
    "o3-mini",
    "claude-4",
    "gemini-2.5-pro",
]

JUDGE_MODEL = "gpt-4o-mini"
JUDGE_PROMPT_FILE = "judge_prompt.txt"

def load_problems(filename="output/all_papers_problems_filtered.json"):
    """Loads problems from the specified JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def get_model_response(model_name, problem_statement):
    """
    Gets a response from the specified LLM.
    """
    try:
        if model_name in ["o3", "o4", "gpt-4o"]:
            params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are an expert in physics. Solve the following problem carefully."},
                    {"role": "user", "content": problem_statement}
                ]
            }
            
            response = openai.chat.completions.create(**params)
            return response.choices[0].message.content
        elif "claude" in model_name.lower():
            # Anthropic API call
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            response_text = ""
            with client.messages.stream(
                model="claude-opus-4-20250514",
                max_tokens=16000,
                thinking={"type": "enabled", "budget_tokens": 8000},
                system="You are an expert in physics. Solve the following problem.",
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
                contents=f"You are an expert in physics. Solve the following problem.\n\n{problem_statement}",
                config=types.GenerateContentConfig(
                    max_output_tokens=4096
                )
            )
            return response.text
    except Exception as e:
        return f"Error getting response: {e}"

def get_judge_evaluation(problem_statement, ground_truth_solution, model_generated_solution):
    """
    Uses an LLM as a judge to evaluate the model's solution.
    """
    with open(JUDGE_PROMPT_FILE, 'r') as f:
        judge_prompt_template = f.read()

    prompt = judge_prompt_template.format(
        problem_statement=problem_statement,
        ground_truth_solution=ground_truth_solution,
        model_generated_solution=model_generated_solution,
    )
    
    try:
        response = openai.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert judge for physics problems."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=1024,
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


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLMs on physics problems.")
    parser.add_argument(
        "--models",
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
        "--output-file",
        type=str,
        default="output/benchmark_results.json",
        help="File to save the benchmark results."
    )
    args = parser.parse_args()

    papers = load_problems()
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

        for model_name in tqdm(args.models, desc=f"Benchmarking models on {problem['paper_id']}", leave=False):
            model_solution = get_model_response(model_name, problem["problem_statement"])
            
            evaluation, score = get_judge_evaluation(
                problem["problem_statement"],
                problem["ground_truth_solution"],
                model_solution
            )
            
            problem_result["model_outputs"][model_name] = {
                "solution": model_solution,
                "evaluation": evaluation,
                "score": score,
            }

        results.append(problem_result)

        # Save incrementally
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=4)

    print(f"Benchmarking complete. Results saved to {args.output_file}")


if __name__ == "__main__":
    main() 