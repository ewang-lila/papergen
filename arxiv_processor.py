import arxiv
import tarfile
import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
import argparse
import json
import glob

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

LLM_PROMPT = """
You are an expert researcher in the sciences. 
## Task  
Create a short set of graduate-level exercise questions, each followed by its solution, based exactly on the published research paper whose tex file is attached. 

---

## Strict requirements  

1. **Independence.** Every question must be *fully self-contained* and exactly follow the steps provided in the paper: embed all notation, assumptions, and context that the student needs *inside that single question*. Do **not** refer to any "previous problem", "same system", "as above", etc.
2. **Focus.** Each question asks the student to *derive* one analytic result that appears in the paper (e.g. a specific equation). Do **not** ask to "show," "prove," or "verify" a result, since the grader only considers the student's final result.
3. **Difficulty balance.** A problem should require a *few* algebraic steps (challenging but doable in \u2264 20 min) and have a unique, objectively checkable answer that is directly written in the paper. I.e., every problem must have a solution that directly corresponds to a result in the paper.
4. **Clarity.** Begin each question with a short "Background" paragraph that defines all symbols and states all assumptions used later in that question, as well as some context. End with a **Task.** sentence that states exactly what the student must show, without revealing the final expression.
5. **Solutions section.** After each question, give the ground-truth solution expression from the paper. For each problem, quote **verbatim** the corresponding equation/result from the paper (with its equation number if present). Do not repeat any derivation steps\u2014just the final expression.
6. **No cross-references.** The wording of one question must not depend on having seen any other question. For example, you may not say anything similar to "Evaluate the integral obtained in Problem\u00a02" or "Using the equation derived in Problem 3."
7. **No extraneous parts** \u2013 Omit numerical verification, coding exercises, open-ended extensions, grading rubrics, etc.
8. **Output the questions and answers only.** Do not include any additional text outside of the question and solutions.
8. **Format everything in Markdown and LaTeX.**

---

### Problem 1  
Background. \u2026  
**Task.** \u2026
**Solution:**
### Problem 2  
Background. \u2026  
**Task.** \u2026
**Solution:**

Following these instructions, read the attached paper and create 3-5 problems according to the aforementioned instructions."""

ARXIV_CATEGORIES = [
    "cond-mat.stat-mech",  # Statistical Mechanics
    "physics.chem-ph",     # Chemical Physics
    "cond-mat.mtrl-sci"   # Condensed Matter Theory (often overlaps with material science)
]

DOWNLOAD_DIR = "arxiv_papers"
OUTPUT_DIR = "output"

def setup_directories():
    """Creates the necessary directories if they don't exist."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Ensured directories '{DOWNLOAD_DIR}' and '{OUTPUT_DIR}' exist.")

def search_and_download_papers(max_results_per_category=5):
    """
    Searches for papers in specified arXiv categories and downloads their source.
    """
    client = arxiv.Client()
    downloaded_files = []

    for category in ARXIV_CATEGORIES:
        print(f"Searching for papers in category: {category}")
        # Search for recent papers in the specified category
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=max_results_per_category,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        for result in client.results(search):
            print(f"Found paper: {result.title} ({result.entry_id})")
            try:
                # Download the source (.tar.gz) file
                filename = result.download_source(dirpath=DOWNLOAD_DIR)
                downloaded_files.append(filename)
                print(f"Downloaded source to: {filename}")
            except Exception as e:
                print(f"Error downloading source for {result.entry_id}: {e}")
    return downloaded_files

def extract_and_combine_tex_files(archive_path):
    """
    Extracts all .tex files from an arXiv source archive and combines them.
    """
    extract_dir = os.path.join(os.path.dirname(archive_path), os.path.basename(archive_path).replace(".tar.gz", "_extracted"))
    os.makedirs(extract_dir, exist_ok=True)
    
    combined_content = ""
    main_tex_path = None # To find the main entry point if needed.
    
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tex_files_members = [m for m in tar.getmembers() if m.name.endswith(".tex")]
            
            # Extract all tex files
            for member in tex_files_members:
                tar.extract(member, path=extract_dir)

            # A common pattern is for a 'main.tex' or similar to include others.
            # We can try to read them in a logical order, but for now, simple concatenation
            # might be sufficient. We can prioritize 'main.tex' to be first.
            
            # Find main.tex and put it first
            main_tex_member = None
            for m in tex_files_members:
                if 'main.tex' in m.name.lower():
                    main_tex_member = m
                    break
            
            if main_tex_member:
                tex_files_members.remove(main_tex_member)
                tex_files_members.insert(0, main_tex_member)

            # Read and concatenate content
            for member in tex_files_members:
                file_path = os.path.join(extract_dir, member.name)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    combined_content += f.read() + "\n\n" # Add separators for clarity

        if combined_content:
            print(f"Successfully extracted and combined {len(tex_files_members)} .tex files from {archive_path}")
            return combined_content
        else:
            print(f"No .tex files found in {archive_path}")

    except tarfile.ReadError:
        print(f"Could not open {archive_path} as a tar.gz file.")
        if archive_path.endswith(".tex"):
            with open(archive_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except Exception as e:
        print(f"Error processing {archive_path}: {e}")
        
    return None

def process_tex_with_llm(tex_content: str):
    """
    Sends the LaTeX content to the OpenAI o3 model for question-answer generation.
    """
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return None

    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model="o3-mini", # Assuming 'o3' is the correct model name
            messages=[
                {"role": "system", "content": LLM_PROMPT},
                {"role": "user", "content": f"Here is the LaTeX content:\n\n{tex_content}"}
            ]
        )
        return response.choices[0].message.content
    except openai.APIStatusError as e:
        print(f"OpenAI API Error: {e.status_code} - {e.response}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred with OpenAI API: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Fetch arXiv papers, process them with an LLM, and save the results.")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip downloading and use existing files in the download directory."
    )
    args = parser.parse_args()

    setup_directories()

    if args.no_download:
        print("Skipping download. Using existing files.")
        downloaded_archives = glob.glob(os.path.join(DOWNLOAD_DIR, "*.tar.gz"))
        if not downloaded_archives:
            print(f"No existing .tar.gz files found in '{DOWNLOAD_DIR}'.")
            return
    else:
        print("Searching for and downloading new papers.")
        downloaded_archives = search_and_download_papers(max_results_per_category=1)

    for archive in downloaded_archives:
        print(f"Processing archive: {archive}")
        combined_tex_content = extract_and_combine_tex_files(archive)
        
        if combined_tex_content:
            print("Sending combined LaTeX content to LLM...")
            llm_output = process_tex_with_llm(combined_tex_content)
            if llm_output:
                # Extract paper ID from the filename for saving
                base_name = os.path.basename(archive)
                # The filename is expected to be like '2506.17126v1.title.tar.gz'
                # We want to get '2506.17126v1'
                id_parts = base_name.split('.')
                paper_id = f"{id_parts[0]}.{id_parts[1]}"
                
                output_data = {
                    "paper_id": paper_id,
                    "llm_output": llm_output
                }
                output_filename = os.path.join(OUTPUT_DIR, f"{paper_id}.json")
                
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=4, ensure_ascii=False)
                
                print(f"LLM output saved to {output_filename}")
            else:
                print("LLM did not return a response or an error occurred.")
        else:
            print(f"Could not find/extract .tex content from {archive}")

if __name__ == "__main__":
    main() 