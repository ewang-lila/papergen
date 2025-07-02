import arxiv
import tarfile
import os
import openai
from openai import OpenAI
from google import genai
from google.genai import types
from dotenv import load_dotenv
import argparse
import json
import glob
import re
from prompt_template import LLM_PROMPT_TEMPLATE

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

num_papers = 15

ARXIV_CATEGORIES = [
    "cond-mat.stat-mech",  # Statistical Mechanics
    # "physics.chem-ph",     # Chemical Physics
    "math-ph",      # Mathematical Physics
    "nlin.ao",
    # "cond-mat.soft",
    # "cond-mat.dis-nn"

]

DOWNLOAD_DIR = "output/arxiv_papers"
OUTPUT_DIR = "output/raw_json_outputs"

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

def parse_llm_output(output_text: str):
    """
    Parses the raw LLM output to extract the problem statement and the boxed solution.
    """
    try:
        solution_marker = "Solution:"
        
        # Check if the essential marker is present
        if solution_marker not in output_text:
            print("Warning: Could not find 'Solution:' marker in LLM output. Unable to parse.")
            return None

        # Split into statement and solution parts
        statement_part, solution_part = output_text.split(solution_marker, 1)

        # Clean up the statement part, removing the original "### Problem" header
        problem_statement = re.sub(r'^\s*### Problem\s*', '', statement_part, flags=re.IGNORECASE).strip()
        
        # Use a greedy regex to capture everything inside \boxed{...}, even across newlines
        match = re.search(r'\\boxed{(.*)}', solution_part, re.DOTALL)
        
        if match:
            final_solution = match.group(1).strip()
        else:
            print("Warning: Could not find a \\boxed{} solution. Using all text after 'Solution:'.")
            final_solution = solution_part.strip()

        # Return None if for some reason the solution is empty
        if not final_solution:
             print("Warning: Parsed solution is empty.")
             return None

        return {"problem_statement": problem_statement, "final_solution": final_solution}
            
    except Exception as e:
        print(f"Error parsing LLM output: {e}")
        return None

def generate_single_problem(tex_content: str, segment: str, problem_number: int):
    """
    Sends the LaTeX content to the OpenAI o3 model to generate a single problem.
    """
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return None

    prompt = LLM_PROMPT_TEMPLATE.format(segment=segment, problem_number=problem_number)
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model="o3", # Assuming 'o3' is the correct model name
            messages=[
                {"role": "system", "content": prompt},
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

def generate_single_problem_gemini(tex_content: str, segment: str, problem_number: int):
    """
    Sends the LaTeX content to the Google Gemini model to generate a single problem.
    """
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        return None

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        
        prompt = LLM_PROMPT_TEMPLATE.format(segment=segment, problem_number=problem_number)
        
        # Combine the prompt and the paper content
        full_prompt = f"{prompt}\\n\\nHere is the LaTeX content:\\n\\n{tex_content}"

        response = client.models.generate_content(
            model="models/gemini-2.5-pro",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.0 # Set deterministic temperature
            ),
        )
        return response.text
    except Exception as e:
        print(f"An unexpected error occurred with Google Gemini API: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Fetch arXiv papers, process them with an LLM, and save the results.")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip downloading and use existing files in the download directory."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of papers to process from the local directory. Only used with --no-download."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini",
        choices=["o3", "gemini"],
        help="The model to use for generation (o3 or gemini)."
    )
    args = parser.parse_args()

    setup_directories()

    if args.no_download:
        print("Skipping download. Using existing files.")
        downloaded_archives = sorted(glob.glob(os.path.join(DOWNLOAD_DIR, "*.tar.gz")))
        if not downloaded_archives:
            print(f"No existing .tar.gz files found in '{DOWNLOAD_DIR}'.")
            return
        if args.limit:
            print(f"Limiting processing to the first {args.limit} papers.")
            downloaded_archives = downloaded_archives[:args.limit]
    else:
        print("Searching for and downloading new papers.")
        downloaded_archives = search_and_download_papers(max_results_per_category=num_papers)

    for archive in downloaded_archives:
        print(f"Processing archive: {archive}")
        combined_tex_content = extract_and_combine_tex_files(archive)
        
        if combined_tex_content:
            problems_data = []
            segments = ["first half", "second half"]
            for i, segment in enumerate(segments):
                print(f"Generating problem {i+1} for the {segment} of the paper using model: {args.model}...")
                
                raw_output = None
                if args.model == "gemini":
                    raw_output = generate_single_problem_gemini(combined_tex_content, segment, i+1)
                else: # Default to o3
                    raw_output = generate_single_problem(combined_tex_content, segment, i+1)

                if raw_output:
                    parsed_problem = parse_llm_output(raw_output)
                    if parsed_problem:
                         # Add problem number to the statement
                        parsed_problem['problem_statement'] = parsed_problem['problem_statement']
                        problems_data.append(parsed_problem)
                    else:
                        print(f"Could not parse LLM output for the {segment} segment.")
                else:
                    print(f"LLM did not return a response for the {segment} segment.")
            
            if problems_data:
                # Combine the individually generated problems into one string
                # Extract paper ID from the filename for saving
                base_name = os.path.basename(archive)
                # The filename is expected to be like '2506.17126v1.title.tar.gz'
                # We want to get '2506.17126v1'
                id_parts = base_name.split('.')
                paper_id = f"{id_parts[0]}.{id_parts[1]}"
                
                output_data = {
                    "paper_id": paper_id,
                    "problems": problems_data
                }
                output_filename = os.path.join(OUTPUT_DIR, f"{paper_id}.json")
                
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=4, ensure_ascii=False)
                
                print(f"All problems saved to {output_filename}")
            else:
                print(f"Could not generate any problems for {archive}")
        else:
            print(f"Could not find/extract .tex content from {archive}")

if __name__ == "__main__":
    main() 