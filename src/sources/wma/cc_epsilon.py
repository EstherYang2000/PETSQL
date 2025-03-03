import json
from wma import WeightedMajorityAlgorithm
from evaluation import evaluate_all_value,build_foreign_key_map_from_json
from sql_gen.call_llm import load_prompts
import argparse
from itertools import zip_longest
import sqlparse
import os
from cc_test import run_sql_generation_wma
import subprocess
import numpy as np


if __name__ == '__main__':

    gold_sql = "./data/spider/dev_gold.sql"
    path_generate = "data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034_3"
    start_num_prompts =  0
    end_num_prompts = 1034
    with open(gold_sql) as f:
        glist = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]
    question_path = os.path.join(path_generate,"questions.json")
    with open(question_path) as f:
        questions = json.load(f)
    print(len(glist),len(questions))
    # all_prompts = load_prompts(args.path_generate,start_num_prompts = args.start_num_prompts,num_prompts=args.num_prompts)

    all_prompts = load_prompts(path_generate,start_num_prompts = start_num_prompts,num_prompts=end_num_prompts)
    input_data = []
    input_data = [{"prompt": prompt, "gold_sql": golden,"question":questions['question']} for prompt, golden,questions in zip_longest(all_prompts, glist[start_num_prompts:end_num_prompts],questions[start_num_prompts:end_num_prompts], fillvalue=None)]
    print(len(input_data))
    # print(input_data[0])
    # # 執行多專家投票 + WMA流程
    accuracy_results = {}   
    epsilon_values = [ round(e,3) for e in np.arange(0.005, 1.0, 0.005).tolist()]

    # epsilon_values = [0.1, 0.2]
    for epsilon in epsilon_values:
        print(f"\n🚀 Running SQL Generation with ε = {epsilon}...")

        # Initialize Weighted Majority Algorithm (WMA) with current epsilon
       
        final_results,results = run_sql_generation_wma(input_data,epsilon,path_generate,start_num_prompts,end_num_prompts)
        # final_path = os.path.join(path_generate,"final_result.json")
        # # 讀取 JSON 檔案
        # with open(final_path, "r", encoding="utf-8") as f:
        #     final_data = json.load(f)
        # Extract `final_sql` values
        final_sql_statements = [entry["final_sql"] for entry in final_results]
        print(final_sql_statements[100])
        # 在寫入文件之前，確保目錄存在
        epsilon_dir = os.path.join(path_generate, "epsilon")
        os.makedirs(epsilon_dir, exist_ok=True)  # 創建 epsilon 目錄（如果不存在）
        output_txt_file = os.path.join(path_generate, "epsilon", f"final_sql_{epsilon}.txt")
        # 然後再寫入文件
        # Write to a text file
        with open(output_txt_file, "w", encoding="utf-8") as f:
            for sql in final_sql_statements:
                f.write(sql + "\n")

        print(f"Successfully written {len(final_sql_statements)} SQL statements to {output_txt_file}")
        
        # Run evaluation script
        print(f"\n📊 Evaluating with ε = {epsilon}...")
        db = "./data/spider/database"
        etype = "all"
        table = "./data/spider/tables.json"
        kmaps = build_foreign_key_map_from_json(table)
        execution_all = evaluate_all_value(gold_sql, output_txt_file, db, etype, kmaps) 
        print(f"Accuracy results with ε = {epsilon}: {execution_all}") 
        accuracy_results[round(float(epsilon), 3)] = execution_all
        

    # Save accuracy results to a JSON file
    accuracy_results_file = os.path.join(path_generate, "epsilon_accuracy_results.json")
    with open(accuracy_results_file, "w", encoding="utf-8") as f:
        json.dump(accuracy_results, f, indent=4)        


        
        