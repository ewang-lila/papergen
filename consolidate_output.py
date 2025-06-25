import json
import glob
import os

OUTPUT_DIR = "output/raw_json_outputs"
CONSOLIDATED_FILENAME = "output/all_papers_problems.json"

def consolidate_json_files():
    """
    Consolidates all individual JSON files from the output directory
    into a single JSON array.
    """
    json_files = glob.glob(os.path.join(OUTPUT_DIR, "*.json"))
    if not json_files:
        print(f"No JSON files found in '{OUTPUT_DIR}' directory to consolidate.")
        return

    all_data = []
    for file_path in sorted(json_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            all_data.append(data)
            print(f"Added data from {os.path.basename(file_path)}.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {file_path}: {e}")

    if all_data:
        with open(CONSOLIDATED_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=4, ensure_ascii=False)
        print(f"Successfully consolidated {len(all_data)} papers into '{CONSOLIDATED_FILENAME}'.")
    else:
        print("No valid data was consolidated.")

if __name__ == "__main__":
    consolidate_json_files() 