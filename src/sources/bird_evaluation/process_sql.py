# import json
# import os
# import argparse

# def extract_sql_candidates(json_file, output_json_file,type):
#     # 1. Read JSON file
#     with open(json_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     # 2. Extract all SQL candidates
#     converted_json = {}
#     for entry in data:
#         idx = None
#         sql_candidates = None
#         if type == 'base':
#             idx = entry["prompt_index"]
#             sql_candidates = entry.get('sql_candidates', "")[0]
#         elif type == 'rf':
#             idx = entry["index"]
#             sql_candidates = entry.get('final_sql', "")
#         converted_json[idx] = sql_candidates
#     # Output the result
#     with open(output_json_file, 'w', encoding='utf-8') as f_out:
#         json.dump(converted_json, f_out, indent=4, ensure_ascii=False)

#     print(f"Extracted SQL candidates and saved to '{output_json_file}'")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--file", default='')  # Path to the input file containing SQL queries
#     parser.add_argument("--output", default="")  # Path to the output file for processed queries
#     parser.add_argument("--type", default='base')
#     args = parser.parse_args()
#     json_file = args.file    # Replace with your actual file path
#     output_txt_file = args.output

#     extract_sql_candidates(json_file, output_txt_file,args.type)

"""
    python bird_evaluation/process_sql.py \
    --file bird/process/bird/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534/llamaapi_3.3_output.json \
    --output bird/process/bird/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534/llamaapi_3.3_output_eval.json
    
# """
import json

# Input and output file paths
input_txt_file = "bird/process/vote/PPL_DEV_ADD_SL_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base_rwma/final_sql_1.txt"
output_json_file = "bird/process/vote/PPL_DEV_ADD_SL_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base_rwma/final_result_1_output_eval.json"

# Read the .txt file and process it into a list
with open(input_txt_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Convert the list into the desired JSON format
converted_json = {}
for idx, line in enumerate(lines):
    converted_json[idx] = line

# Write the result to the output JSON file
with open(output_json_file, 'w', encoding='utf-8') as f_out:
    json.dump(converted_json, f_out, indent=4, ensure_ascii=False)

print(f"Converted data has been saved to '{output_json_file}'")

