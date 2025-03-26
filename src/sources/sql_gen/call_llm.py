"""

根據已生成的 prompts（如 generated_prompts.txt），呼叫對應 LLM ，
產生最終 SQL 結果，並寫到 out_file。
"""
print(f"[DEBUG] 正在執行的 call_llm.py = {__file__}")

import os
import argparse
from time import sleep
from transformers import pipeline
from tqdm import tqdm
import json
import sqlparse
from llms import SQLCoder, vicuna, GPT,OllamaChat,GroqChat

from post_process import extract_sql
from refine.refinement import run_refinement_pipeline
from sql_utils import run_sql_generation, run_refinement_pipeline
from utils.file_utils import load_prompts
# 你的專案內部匯入

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 全域 pipeline (若需 huggingface pipeline)
model_pipeline = None




if __name__ == '__main__':
    # 创建 ArgumentParser 对象

    parser = argparse.ArgumentParser(description="Call LLM on prompts and output results.")

    parser.add_argument("--model", type=str, default="llamaapi",
                        help="Which model to use? codellamaapi, puyuapi, llamaapi, sqlcoderapi, vicunaapi, gptapi")
    parser.add_argument("--model_version", type=str, default="none",
                        help="Which GPT version to use with gptapi? Options: o1-preview, gpt-4, gpt-4o")
    parser.add_argument("--path_generate", type=str,
                        help="Path to the generated raw file.")
    parser.add_argument("--dataset_type", type=str,
                        help="")
    parser.add_argument("--out_file", type=str, default="raw.txt")
    # parser.add_argument("--pool", type=int, default=1)
    parser.add_argument("--start_num_prompts", type=int, default=0)
    parser.add_argument("--end_num_prompts", type=int, default=1034,
                        help="Number of prompts to process from the prompt file (if not specified, take all).")
    parser.add_argument("--call_mode", type=str, default="write",
                        help="mode to write or append")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="")
    parser.add_argument("--n_samples", type=int, default=1,
                        help="Number of samples to generate for each prompt.")
    parser.add_argument("--refinement", action="store_true",
                    help="whether to do refinement")


    args = parser.parse_args()
    print(args)
    # 根據 dataset 及其他參數組成 path_generate (這裡只示範)
    all_prompts = load_prompts(args.path_generate,start_num_prompts = args.start_num_prompts,end_num_prompts=args.end_num_prompts)
    print(len(all_prompts))
    
    question_path = os.path.join(args.path_generate,"questions.json")
    with open(question_path) as f:
        questions = json.load(f)
    input_questions = questions[args.start_num_prompts:args.end_num_prompts]
    # 執行主程式
    sql_candidates = run_sql_generation(
        model=args.model,
        path_generate = args.path_generate,
        prompts = all_prompts,
        out_file=args.out_file,
        model_version=args.model_version,
        start_num_prompts = args.start_num_prompts,
        end_num_prompts=args.end_num_prompts,
        call_mode = args.call_mode,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
    )
    if args.dataset_type == "dev":
        database_path = "./data/spider/database/" 
    else:
        database_path = "./data/spider/test_database/" 

    if args.refinement:
        
        refined_sql_candidates = []
        for index,sql_entry in enumerate(sql_candidates):  # 逐一處理每個 prompt 的 SQL
            db_id = input_questions[index]['db']
            db_path = f"{database_path}{db_id}/{db_id}.sqlite"
            
            refined_sql = run_refinement_pipeline(
                db_path, all_prompts[index], [sql_entry], args.path_generate, args.end_num_prompts, args.model
            )
            refined_sql_candidates.append(refined_sql)

        # 儲存 refined SQL
        refined_out_file = os.path.join(args.path_generate, f"{args.model}_refined.json")
        with open(refined_out_file, "w", encoding="utf-8") as f_out:
            json.dump(refined_sql_candidates, f_out, indent=4, ensure_ascii=False)

        print(f"Done. Refined SQLs saved to {refined_out_file}")

"""
python src/sources/sql_gen/call_llm.py \
    --path_generate data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034_dev3 \
    --model llamaapi \
    --model_version 3.3 \
    --out_file llamaapi_32b_3.3.json \
    --dataset_type dev \
    --start_num_prompts 0 \
    --end_num_prompts 1 \
    --batch_size 1 \
    --call_mode append \
    --n_samples 1 \
    --refinement

"""

