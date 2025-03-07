import os
import json

def load_prompts(path_generate,start_num_prompts = None,end_num_prompts=None):
    # 1) 讀取 prompt_file
    prompt_path = os.path.join(path_generate, "prompts.txt")

    with open(prompt_path, "r", encoding='utf-8') as f_in:
        content = f_in.read()

    # 以 "\n\n" 為分隔符，分割出所有 prompt
    all_prompts = content.split("\n\n\n\n")

    # 如果有指定要處理的 prompt 數量，就截斷
    if end_num_prompts is not None and start_num_prompts is not None and start_num_prompts >=0 and end_num_prompts > 0:
        all_prompts = all_prompts[start_num_prompts:end_num_prompts]
    print(f"Loaded {len(all_prompts)} prompts.")
    return all_prompts


def append_json(file_path, new_data):
    """ Append new data to an existing JSON file or create a new one if it doesn't exist. """
    # Check if file exists and read existing content
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]  # Ensure it's a list
            except json.JSONDecodeError:
                existing_data = []  # If file is empty or corrupt, start fresh
    else:
        existing_data = []

    # Append new data
    if isinstance(new_data, list):
        existing_data.extend(new_data)
    else:
        existing_data.append(new_data)

    # Write back to file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)

def read_json(file_path:str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
def write_txt(file_path:str,data):
    # Write to a text file
    with open(file_path, "w", encoding="utf-8") as f:
        for sql in data:
            f.write(sql + "\n")
