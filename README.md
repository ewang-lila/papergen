The project can be run via 
```bash
python arxiv_processor.py
```

This will download the most recent arXiv paper from each of the categories listed in the file. Adding the 
```--no-download``` flag will process only the files already downloaded locally. The default model is Gemini 2.5 Flash, but o3 is also an option. 

Running 
```bash
python arxiv_processor.py --model gemini --no-download
```
will 

The problems and answers can be rendered in your browser using 
```bash
python render_output.py && open output.html
```
This doesn't work very well though, so it's more reliable to convert the jsons to LaTeX and manually compile in overleaf using
```bash
python export_to_tex.py                               
```

## Benchmarking LLMs

This project also includes a script to benchmark various LLMs on the problems generated from the papers.

### Setup

1.  **Install dependencies:** Make sure you have all the required packages installed.
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set up API Keys:** Create a `.env` file in the root of the project and add your API keys for the services you want to use.

    ```
    OPENAI_API_KEY="your_openai_api_key"
    ANTHROPIC_API_KEY="your_anthropic_api_key"
    GOOGLE_API_KEY="your_google_api_key"
    ```

### Running the Benchmark

You can run the benchmark using the `benchmark_llms.py` script.

```bash
python benchmark_llms.py
```

The script supports several command-line arguments:

*   `--models`: A list of models to benchmark. Defaults to all supported models.
    Example: `python benchmark_llms.py --models gpt-4o claude-3-opus-20240229`
*   `--limit`: Limit the number of problems to evaluate for a quick test.
    Example: `python benchmark_llms.py --limit 5`
*   `--output-file`: Specify a custom output file for the results.
    Example: `python benchmark_llms.py --output-file output/my_benchmark_results.json`

The results, including the model-generated solutions, judge's evaluation, and scores, will be saved to `output/benchmark_results.json` by default.