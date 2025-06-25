import json
import glob
import os

# Directories
INPUT_FILENAME = "output/all_papers_problems.json"
EXPORT_DIR = "output/tex_exports"

# LaTeX document template
TEX_TEMPLATE = r"""
\documentclass{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{amsthm}}
\usepackage{{amsfonts}}
\usepackage{{graphicx}}
\usepackage{{xcolor}}

\title{{{title}}}
\author{{From arXiv:{paper_id}}}
\date{{\today}}

\begin{{document}}

\maketitle

{content}

\end{{document}}
"""

def export_to_tex():
    """
    Generates a .tex file for each JSON output, for easier inspection of LaTeX.
    """
    # Ensure the export directory exists
    os.makedirs(EXPORT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_FILENAME):
        print(f"Input file '{INPUT_FILENAME}' not found. Please run consolidate_output.py first.")
        return

    with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
        all_papers_data = json.load(f)
    
    if not isinstance(all_papers_data, list):
        print(f"Expected a list of papers in '{INPUT_FILENAME}', but found a different structure.")
        return

    exported_files = []
    for paper_data in all_papers_data:
        paper_id = paper_data.get("paper_id", "Unknown")
        problems = paper_data.get("problems", [])
        
        tex_content = ""
        for i, problem in enumerate(problems):
            problem_statement = problem.get("problem_statement", "")
            final_solution = problem.get('final_solution', '')

            # Add problem statement
            tex_content += f"%% --- Problem {i+1} ---\n\n"
            tex_content += problem_statement

            # Add solution
            tex_content += f"\n\n\\textbf{{Solution:}}\n\n"
            tex_content += f"\\boxed{{{final_solution}}}\n\n\\newpage\n\n"

        # Populate the main TeX template
        final_tex = TEX_TEMPLATE.format(
            title=f"Problems for Paper {paper_id}",
            paper_id=paper_id,
            content=tex_content
        )
        
        # Write to file
        output_filename = os.path.join(EXPORT_DIR, f"{paper_id}.tex")
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(final_tex)
        exported_files.append(output_filename)

    if exported_files:
        print(f"Successfully exported {len(exported_files)} .tex files to the '{EXPORT_DIR}' directory.")
        for fname in exported_files:
            print(f"- {fname}")
    else:
        print("No files were exported.")

if __name__ == "__main__":
    export_to_tex() 