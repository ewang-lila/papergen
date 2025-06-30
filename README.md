This project contains a suite of tools to download physics papers from arXiv, convert their key findings into solvable problems, evaluate LLMs against these problems, and generate benchmark reports.

## Core Workflow

The project is designed to be run as a sequential pipeline. Here is the recommended workflow:

### Step 1: Generate Problems from Papers
First, download papers from arXiv and use an LLM to generate problems from their content. The script saves one JSON file per paper in `output/raw_json_outputs/`.

```bash
python arxiv_processor.py
```

**Common Options:**
- `--no-download`: Use this flag to skip downloading and process papers already present in `output/arxiv_papers/`.
- `--limit <N>`: When used with `--no-download`, this will only process the first `N` papers found locally.
- `--model <model_name>`: Choose the model for problem generation (e.g., `gemini` or `o3`).

Example: Process the first 5 local papers using the `gemini` model.
```bash
python arxiv_processor.py --no-download --model gemini --limit 5
```

### Step 2: Consolidate and Filter Raw Outputs
Next, combine the many individual JSON files into a single, clean master file. This script also filters out low-quality or malformed problems and reports on how many were removed.

```bash
python consolidate_and_filter.py
```
This script reads all files from `output/raw_json_outputs/`, filters them, and creates a single clean dataset at `output/all_papers_problems_filtered.json`.

### Step 3: Benchmark LLMs
Evaluate the performance of different LLMs on the set of filtered problems.

```bash
python benchmark_llms.py
```
This script takes the filtered problems, gets solutions from the specified LLMs, uses a judge model to score them, and saves the detailed results to `output/benchmark_results.json`.

**Common Options:**
*   `--models`: Specify which models to benchmark (e.g., `o3`, `gpt-4o`).
*   `--limit`: Limit the number of problems to evaluate for a quick test.

Example: Benchmark `o3` on the first 10 problems.
```bash
python benchmark_llms.py --models o3 --limit 10
```

### Step 4: Generate a LaTeX Report
Create a high-quality, human-readable report from the benchmark results.

```bash
python export_benchmark_to_tex.py
```
This script reads `output/benchmark_results.json` and generates a comprehensive LaTeX file at `output/solutions_report.tex`. You can then compile this into a PDF using any LaTeX distribution (e.g., `pdflatex`):
```bash
pdflatex -output-directory=output output/solutions_report.tex
```

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

### Now important: run the refinement agent

Default: 
```bash
python refine_problems.py
```
Specifying how much input data file to use: 
```bash
python refine_problems.py --max-problems 10
```

Note that this will also create a json file with the critiques and updated problems at outputs/critiques. To do the benchmarking using the revised problems:
```bash
python benchmark_llms.py --input-file output/revised_problems.json --output-file output/revised_benchmark_results.json
```

Then, export the revised benchmark results to LaTeX
```bash
python export_benchmark_to_tex.py --benchmark-results-file output/revised_benchmark_results.json --output-tex-file output/revised_solutions_report.tex
```

Using the default commands from above will use all_papers_problems_filtered

### Saving the critiques

Use 
```bash
python export_critiques_to_tex.py
```
