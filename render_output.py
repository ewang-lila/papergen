import json
import glob
import os
import markdown2

OUTPUT_DIR = "output"
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>arXiv Q&A Output</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }}
        h1 {{
            color: #1a1a1a;
            text-align: center;
            border-bottom: 2px solid #eaeaea;
            padding-bottom: 10px;
        }}
        .paper {{
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .problem {{
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid #f0f0f0;
        }}
        .problem:last-child {{
            border-bottom: none;
        }}
        h2 {{
            color: #0056b3;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
        }}
        code {{
            background-color: #eee;
            padding: 2px 4px;
            border-radius: 4px;
        }}
        pre code {{
            display: block;
            padding: 10px;
            border-radius: 4px;
            white-space: pre-wrap;
        }}
    </style>
</head>
<body>
    <h1>arXiv Question-Answer Pairs</h1>
    {content}
</body>
</html>
"""

PAPER_TEMPLATE = """
<div class="paper">
    <h2>Paper: <a href="https://arxiv.org/abs/{paper_id}" target="_blank">{paper_id}</a></h2>
    <div class="problems-container">
        {paper_content}
    </div>
</div>
"""

def render_html():
    """Generates an HTML file from the JSON outputs."""
    json_files = glob.glob(os.path.join(OUTPUT_DIR, "*.json"))
    if not json_files:
        print(f"No JSON files found in '{OUTPUT_DIR}' directory.")
        return

    all_papers_html = ""
    for file_path in sorted(json_files):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        paper_id = data.get("paper_id", "Unknown")
        problems = data.get("problems", [])
        
        paper_content_html = ""
        for problem in problems:
            statement_html = markdown2.markdown(
                problem.get("problem_statement", ""), 
                extras=["fenced-code-blocks", "code-friendly"]
            )
            # Wrap the LaTeX solution for MathJax rendering as a block equation
            final_solution = problem.get('final_solution', '')
            solution_html = f"<strong>Solution:</strong>\n\\[{final_solution}\\]"
            
            paper_content_html += f"<div class='problem'>{statement_html}\n{solution_html}</div>"

        paper_html = PAPER_TEMPLATE.format(paper_id=paper_id, paper_content=paper_content_html)
        all_papers_html += paper_html

    final_html = HTML_TEMPLATE.format(content=all_papers_html)
    
    output_filename = "output.html"
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(final_html)
        
    print(f"Successfully rendered output to '{output_filename}'. Open this file in your browser.")

if __name__ == "__main__":
    render_html() 