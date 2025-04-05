import json
import argparse

def extract_sql_candidates(json_file, output_txt_file):
    # 1. Read JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Extract all SQL candidates
    all_sqls = []
    for entry in data:
        sql_candidates = entry.get('sql_candidates', [])
        all_sqls.extend(sql_candidates)
    # 3. Write to TXT file
    with open(output_txt_file, 'w', encoding='utf-8') as f_out:
        for sql in all_sqls:
            f_out.write(sql.strip() + "\n")

    print(f"Extracted {len(all_sqls)} SQL candidates and saved to '{output_txt_file}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default='')  # Path to the input file containing SQL queries
    parser.add_argument("--output", default="")  # Path to the output file for processed queries
    args = parser.parse_args()
    json_file = args.file    # Replace with your actual file path
    output_txt_file = args.output

    extract_sql_candidates(json_file, output_txt_file)


