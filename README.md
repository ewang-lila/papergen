This project contains a suite of tools to download physics papers from arXiv, convert their key findings into solvable problems, evaluate LLMs against these problems, and generate benchmark reports.

## Core Workflow

The project is designed to be run as a sequential pipeline. Here is the recommended workflow:

### Step 1: Generate Problems from Papers
First, download papers from arXiv and use an LLM to generate problems from their content. The script saves one JSON file per paper in `output/papers/initial_QA_pairs/` and downloads raw arXiv papers to `output/papers/arxiv_papers/`.

```bash
python scripts/generation/arxiv_processor.py
```

**Common Options:**
- `--no-download`: Use this flag to skip downloading and process papers already present in `output/papers/arxiv_papers/`.
- `--limit <N>`: When used with `--no-download`, this will only process the first `N` papers found locally.
- `--model <model_name>`: Choose the model for problem generation (e.g., `gemini` or `o3`).
- `--workers <N>`: Process papers in parallel using `N` worker threads.

Example: Process the first 5 local papers using the `gemini` model with 5 workers.
```bash
python scripts/generation/arxiv_processor.py --no-download --model gemini --limit 5 --workers 5
```

### Step 2: Consolidate and Filter Raw Outputs
Next, combine the many individual JSON files into a single, clean master file. This script also filters out low-quality or malformed problems and reports on how many were removed.

```bash
python scripts/generation/consolidate_and_filter.py
```
This script reads all files from `output/papers/initial_QA_pairs/`, filters them, and creates a single clean dataset at `output/problems/all_papers_problems_filtered.json`.

### Step 3: Refine Problems with a Critic Agent
Refine the generated problems for self-containment and difficulty using a critic agent. This step also generates critiques and revised problems.

Default: 
```bash
python scripts/generation/refine_problems.py
```
Specifying how much input data file to use: 
```bash
python scripts/generation/refine_problems.py --max-problems 10
```

This will create a JSON file with the critiques at `output/critiques/all_critiques.json` and updated problems at `output/problems/refined_problems.json`.

### Step 4: Benchmark LLMs
Evaluate the performance of different LLMs on the set of filtered or refined problems.

```bash
python scripts/evals/benchmark_llms.py
```
This script takes the filtered problems (by default from `output/problems/refined_problems.json`), gets solutions from the specified LLMs, uses a judge model to score them, and saves the detailed results to `output/results/benchmark_results_{model_name}.json` or `output/results/benchmark_results.json`.

**Common Options:**
*   `--models`: Specify which models to benchmark (e.g., `o3`, `gpt-4o`).
*   `--limit`: Limit the number of problems to evaluate for a quick test.
*   `--input-file`: Specify an alternative input problem file, e.g., `output/problems/all_papers_problems_filtered.json`.

Example: Benchmark `o3` on the first 10 problems using revised problems.
```bash
python scripts/evals/benchmark_llms.py --models o3 --limit 10 --input-file output/problems/refined_problems.json
```

### Step 5: Generate a LaTeX Report for Solutions
Create a high-quality, human-readable report from the benchmark results.

```bash
python scripts/evals/export_benchmark_to_tex.py --model <model_name>
```
This script reads `output/results/{model_name}/benchmark_results_{model_name}.json` and generates a comprehensive LaTeX file at `output/results/{model_name}/tex/solutions_report_{model_name}.tex`. The report is split into multiple subfiles (e.g., `problems_part_1.tex`) in a `output/results/{model_name}/tex/problem_parts/` subdirectory, which are then included in the main report.

You can then compile the main `.tex` file into a PDF using any LaTeX distribution (e.g., `pdflatex`):
```bash
pdflatex -output-directory=output/results/{model_name}/pdf output/results/{model_name}/tex/solutions_report_{model_name}.tex
```

### Step 6: Generate a LaTeX Report for Critiques
Generate a LaTeX report detailing the critiques from the problem refinement process.

```bash
python scripts/evals/export_critiques_to_tex.py
```
This script reads `output/critiques/all_critiques.json` and generates a LaTeX file at `output/critiques/critiques_report.tex`.

## Directory Structure

-   `flyte/`: Contains Flyte workflow definitions.
-   `scripts/`:
    -   `generation/`: Scripts for data acquisition and problem generation.
        -   `arxiv_processor.py`
        -   `consolidate_and_filter.py`
        -   `generate_solution_traces.py`
        -   `refine_problems.py`
        -   `prompt_template.py`
    -   `evals/`: Scripts for LLM benchmarking and report generation.
        -   `benchmark_llms.py`
        -   `export_benchmark_to_tex.py`
        -   `export_correct_problems.py`
        -   `export_critiques_to_tex.py`
        -   `judge_prompt.txt`
-   `output/`:
    -   `papers/`: Contains raw data from arXiv.
        -   `arxiv_papers/`: Downloaded `.tar.gz` files.
        -   `initial_QA_pairs/`: Raw JSON files with problems generated by `arxiv_processor.py`.
    -   `problems/`: Stores processed problem sets.
        -   `all_papers_problems_filtered.json`: Consolidated and filtered problems from `consolidate_and_filter.py`.
        -   `refined_problems.json`: Problems refined by `refine_problems.py`.
    -   `critiques/`: Contains all files related to the problem refinement process.
        -   `all_critiques.json`: Consolidated critiques from `refine_problems.py`.
        -   `debug/`: Debugging information for critiques.
        -   `critiques_report.tex`: LaTeX report of critiques generated by `export_critiques_to_tex.py`.
    -   `results/`: Stores benchmark results and solution reports.
        -   `benchmark_results.json` / `benchmark_results_{model_name}.json`: Benchmark results from `benchmark_llms.py`.
        -   `solutions_report.tex` / `solutions_report_{model_name}.tex`: Main LaTeX report for solutions generated by `export_benchmark_to_tex.py`.
        -   `problems_{model_name}/`: Subdirectory containing split LaTeX files for individual problems.
    -   `tex_outputs/`: Where compiled PDF outputs of LaTeX documents should be stored.

## Setup for Benchmarking

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set up API Keys:** Create a `.env` file in the project root and add your API keys:
    ```
    OPENAI_API_KEY="your_openai_api_key"
    ANTHROPIC_API_KEY="your_anthropic_api_key"
    GOOGLE_API_KEY="your_google_api_key"
    ```
