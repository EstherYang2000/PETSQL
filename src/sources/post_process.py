import re, os
import sqlparse
import argparse
import json
from utils.post_process import extract_sql



if __name__ == "__main__":
    # Set up argument parsing for command-line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default='codellama')  # Specify the LLM type (default: codellama)
    parser.add_argument("--file", default='')  # Path to the input file containing SQL queries
    parser.add_argument("--output", default="")  # Path to the output file for processed queries


    args = parser.parse_args()

    llm = args.llm
    file_ext = os.path.splitext(args.file)[-1].lower()

    if file_ext == ".json":
        print(f"Processing JSON file: {args.file}")
        # 讀JSON檔案
        with open(args.file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        # 每個prompt的SQL candidates處理
        for index,entry in enumerate(data):
            processed_candidates = []
            if entry['sql_candidates']:  
                if isinstance(entry['sql_candidates'], list):
                    
                    for sql in entry['sql_candidates']:
                        extracted_sql = sqlparse.format(extract_sql(sql, llm=args.llm), reindent=False)
                        # print(f"Extracted SQL {index}: {extracted_sql}")  # Debugging log
                        # formatted_list = [sqlparse.format(extracted_sql, reindent=False) for extracted_sql in extracted_list]
                        processed_candidates.append(extracted_sql)
                else:
                    formatted_query = sqlparse.format(entry['sql_candidates'].strip(), reindent=False)
                    # extracted_query.append(formatted_query.replace(" '''",""))
                    # extracted_sql = extract_sql(entry['sql_candidates'], llm=args.llm)
                    # formatted_sql = sqlparse.format(extracted_sql, reindent=False)
                    processed_candidates.append(formatted_query.replace(" '''",""))
            else:
                print(f"Extracted SQL {index}: {entry}")  # Debugging log
            print(f"Processed SQL candidates: {processed_candidates}")
            entry['sql_candidates'] = processed_candidates
            # 存回新的JSON檔案
        output_path = args.output if args.output else args.file.replace(".json", "_processed.json")
        
        
        with open(output_path, 'w', encoding='utf-8') as f_out:
            json.dump(data, f_out, indent=4, ensure_ascii=False)

        print(f"Processing complete. Results saved to: {output_path}")
    elif file_ext == ".txt":
        
        # 讀檔案
        with open(args.file, 'r', encoding='utf-8') as file:
            content = file.readlines()
        extracted_query = []
        for index,q in enumerate(content):
            result = extract_sql(q, "sensechat")
            if not result:  
            #     print(f"Extracted SQL {index}: {result}")  # Debugging log
            # else:
                print(f"Failed to extract SQL from line {index}:")  # Log missing SQL
            if result:  # Check if result is not None or empty
                if isinstance(result, list):  # Handle multiple queries
                    formatted_queries = [sqlparse.format(query.strip(), reindent=False) if query else None for query in result]
                    extracted_query.extend(formatted_queries)
                else:  # Single query
                    formatted_query = sqlparse.format(result.strip(), reindent=False)
                    extracted_query.append(formatted_query.replace(" '''",""))
            else:
                extracted_query.append("")

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as file:
                file.write('\n'.join(extracted_query))
        else:
            with open(args.file.replace(".txt", "_out.txt") , 'w', encoding='utf-8') as file:
                file.write('\n'.join(extracted_query))


#python src/sources/post_process.py --file src/sources/raw_codellama.txt --output src/sources/output_codellama.txt --llm codellama
#python src/sources/post_process.py --file src/sources/raw_gpt.txt --output src/sources/output_gpt.txt --llm gpt
# python src/sources/post_process.py --file data/process/PPL_TEST.JSON-9_SHOT_Euclidean_mask_1034/qwen_api_2_5_72b_test.json --output data/process/PPL_TEST.JSON-9_SHOT_Euclidean_mask_1034/qwen_api_2_5_72b_test_output.json --llm sensechat