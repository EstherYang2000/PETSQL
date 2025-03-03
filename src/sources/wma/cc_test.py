import json
from wma import WeightedMajorityAlgorithm
from post_process import extract_sql  # 假設在 post_process.py 中定義
from evaluation import evaluate_cc,build_foreign_key_map_from_json
from sql_gen.call_llm import run_sql_generation,load_prompts
import argparse
from itertools import zip_longest
import sqlparse
import os
# 初始化 Weighted Majority Algorithm



def load_sql_from_txt(file_path):
    """從 TXT 文件中載入 SQL 結果，每行代表一個 SQL 預測"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]
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

def call_expert(expert_name: str, prompt:str,path_generate:str,end_num_prompts=1034,model_version=None,) -> str:
    """
    模擬「呼叫對應模型」並回傳 (可能帶雜訊的) raw SQL。
    實務中應依照各自 LLM/模型的呼叫流程實作。
    
    :param expert_name: 模型/專家名稱 (e.g. "codellamaapi")
    :param sample: 包含問題等資訊的字典，例如 {"question": "...", "gold_sql": "..."}
    :return: raw SQL（模型輸出的原始字串）
    """
    # question = sample['question']
    # 在真實應用中，應將 question 塞入 prompt，呼叫 LLM API。
    # 這裡示範用 if-else 模擬不同模型輸出。
    
    if expert_name == "codellamaapi":
        return run_sql_generation(model=expert_name,prompts=[prompt],path_generate=path_generate ,out_file="codellama.txt",num_prompts=end_num_prompts,pool_num=1,call_mode="append")
    elif expert_name == "puyuapi":
        return run_sql_generation(model=expert_name,prompts=[prompt],path_generate=path_generate, out_file="puyuma.txt",num_prompts=end_num_prompts,pool_num=1,call_mode="append")
    elif expert_name == "gptapi":
        if model_version == "gpt-3.5-turbo":
            return run_sql_generation(model=expert_name,prompts=[prompt], path_generate=path_generate,out_file=f"{expert_name}_{model_version}_cc.txt",num_prompts=end_num_prompts,pool_num=1,call_mode="append",model_version=model_version)
        elif model_version == "gpt-4":
            return run_sql_generation(model=expert_name,prompts=[prompt], path_generate=path_generate,out_file=f"{expert_name}_{model_version}_cc.txt",num_prompts=end_num_prompts,pool_num=1,call_mode="append",model_version=model_version)
        elif model_version == "gpt-4o":
            return run_sql_generation(model=expert_name,prompts=[prompt], path_generate=path_generate,out_file=f"{expert_name}_{model_version}_cc.txt",num_prompts=end_num_prompts,pool_num=1,call_mode="append",model_version=model_version)
        elif model_version == "o1-preview":
            return run_sql_generation(model=expert_name,prompts=[prompt], path_generate=path_generate,out_file=f"{expert_name}_{model_version}_cc.txt",num_prompts=end_num_prompts,pool_num=1,call_mode="append",model_version=model_version)
    elif expert_name == "qwen_api":
        if model_version == "32b-instruct-fp16":
            return run_sql_generation(model=expert_name,prompts=[prompt], path_generate=path_generate,out_file=f"{expert_name}_{model_version}_cc.txt",num_prompts=end_num_prompts,pool_num=1,call_mode="append",model_version=model_version)
    elif expert_name == "llamaapi":
        if model_version == "3.3":
            return run_sql_generation(model=expert_name,prompts=[prompt], path_generate=path_generate,out_file=f"{expert_name}_{model_version}_cc.txt",num_prompts=end_num_prompts,pool_num=1,call_mode="append",model_version=model_version)
        elif model_version == "3.3_70b_specdec":
            return run_sql_generation(model=expert_name,prompts=[prompt], path_generate=path_generate,out_file=f"{expert_name}_{model_version}_cc.txt",num_prompts=end_num_prompts,pool_num=1,call_mode="append",model_version=model_version)
    elif expert_name == "deepseekapi":
        if model_version == "r1_70b":
            return run_sql_generation(model=expert_name,prompts=[prompt], path_generate=path_generate,out_file=f"{expert_name}_{model_version}_cc.txt",num_prompts=end_num_prompts,pool_num=1,call_mode="append",model_version=model_version)
        elif model_version == "v2-16b":
            return run_sql_generation(model=expert_name,prompts=[prompt], path_generate=path_generate,out_file=f"{expert_name}_{model_version}_cc.txt",num_prompts=end_num_prompts,pool_num=1,call_mode="append",model_version=model_version)
        return run_sql_generation(model=expert_name,prompts=[prompt], path_generate=path_generate,out_file=f"{expert_name}_{model_version}_cc.txt",num_prompts=end_num_prompts,pool_num=1,call_mode="append",model_version=model_version)
    else:
        return run_sql_generation(model=expert_name,prompts=[prompt], path_generate=path_generate,out_file="model.txt",num_prompts=end_num_prompts,pool_num=1,call_mode="append")
"""
run_sql_generation(
        model=args.model,
        path_generate = args.path_generate,
        prompts = all_prompts,
        out_file=args.out_file,
        pool_num=args.pool,
        model_version=args.model_version,
        num_prompts=args.num_prompts,
        call_mode = args.call_mode,
        batch_size=args.batch_size
    )
python src/sources/sql_gen/call_llm.py \
    --path_generate /home/yyj/Desktop/yyj/thesis/code/PETSQL/data/process/PPL_DEV.JSON-3_SHOT_Euclidean_mask_1034 \
    --model deepseekapi \
    --model_version r1_70b \
    --out_file deepseek_r1_70b_api.txt \
    --num_prompts 1034 \
    --batch_size 1
"""
def run_sql_generation_wma(input_data,epsilon,path_generate,start_num_prompts,end_num_prompts,cc = True):
    """
    結合 WeightedMajorityAlgorithm (WMA) 與多位專家模型輸出，最終投票產生 SQL。

    :param input_data: List[dict]，其中每個 dict 至少包含:
        {
            "question": str,     # 問題
            "gold_sql": str      # 正解 SQL (可選，若要判斷正確性/更新權重)
        }
    :return: List[dict]，每個元素代表一筆資料的處理結果，包含:
        {
            "question": str,
            "raw_sql_codellama": str,
            "raw_sql_puyu": str,
            "raw_sql_llama": str,
            "final_sql": str,       # 投票+清洗後
            "gold_sql": str,
            "is_correct": bool,
            "chosen_experts": List[str],
            "current_weights": Dict[str, float]
        }
    """

    wma = WeightedMajorityAlgorithm(epsilon=epsilon)
    wma.add_expert("llamaapi_3.3", init_weight=1.0)
    wma.add_expert("qwen2.5-coderaapi", init_weight=1.0)
    wma.add_expert("mistralapi", init_weight=1.0)
    wma.add_expert("qwen2.5_72b", init_weight=1.0)

    
    # Add validation
    if not input_data:
        raise ValueError("input_data cannot be empty")
        
    for sample in input_data:
        if 'prompt' not in sample or not sample['prompt']:
            raise ValueError(f"Invalid sample, missing or empty prompt: {sample}")
    results = []
    final_results = []
    # sql_gpt4 = load_sql_from_txt(os.path.join(path_generate, "gptapi_gpt-4_cc.txt"))
    # sql_gpt4o = load_sql_from_txt(os.path.join(path_generate, "gptapi_gpt-4o_cc.txt"))
    # sql_gpto1preview = load_sql_from_txt(os.path.join(path_generate, "gptapi_o1-preview_cc.txt"))
    # sql_o3mini = load_sql_from_txt(os.path.join(path_generate, "gptapi_o3-mini_cc.txt"))
    sql_llama3_3 = load_sql_from_txt(os.path.join(path_generate, "llamaapi_3.3_cc.txt"))
    sql_qwen2_5 = load_sql_from_txt(os.path.join(path_generate, "qwen_api_32b-instruct-fp16_cc.txt"))
    # sql_deepseek_r1_70b = load_sql_from_txt(os.path.join(path_generate, "deepseekapi_r1_distill_llama_70b_cc.txt")) 
    sql_mistral = load_sql_from_txt(os.path.join(path_generate, "mistralapi_small_24b_cc.txt"))
    sql_qwen2_5_72b = load_sql_from_txt(os.path.join(path_generate, "qwen2_5_72b_cc.txt"))
    for index,sample in enumerate(input_data):
        print(f"-------------Processing sample {index+start_num_prompts}...-------------")
        
        # B. 每位專家的 raw SQL 都先做基礎清洗

        # clean_sql_gpt4 = sqlparse.format(extract_sql(sql_gpt4[index], "sensechat").strip(), reindent=False)
        # clean_sql_gpt4o = sqlparse.format(extract_sql(sql_gpt4o[index], "sensechat").strip(), reindent=False)
        # clean_sql_gpto1preview = sqlparse.format(extract_sql(sql_gpto1preview[index], "sensechat").strip(), reindent=False)
        # clean_o3mini = sqlparse.format(extract_sql(sql_o3mini[index], "sensechat").strip(), reindent=False)
        clean_sql_llama3_3 = sqlparse.format(extract_sql(sql_llama3_3[index], "sensechat").strip(), reindent=False)
        clean_sql_qwen2_5 = sqlparse.format(extract_sql(sql_qwen2_5[index], "sensechat").strip(), reindent=False)
        # clean_sql_deepseek_r1_70b = sqlparse.format(extract_sql(sql_deepseek_r1_70b[index], "sensechat").strip(), reindent=False)
        clean_sql_mistral = sqlparse.format(extract_sql(sql_mistral[index], "sensechat").strip(), reindent=False)
        clean_sql_qwen2_5_72b = sqlparse.format(extract_sql(sql_qwen2_5_72b[index], "sensechat").strip(), reindent=False)


        # C. 建立投票用 predictions dict
        predictions = {
            # "gptapi4":      clean_sql_gpt4,
            # "gptapi4o":      clean_sql_gpt4o,
            # "o1-preview":      clean_sql_gpto1preview,
            # "o3-mini":      clean_o3mini,
            "llamaapi_3.3": clean_sql_llama3_3,
            "qwen2.5-coderaapi": clean_sql_qwen2_5,
            # "deepseekapi_r1_70b": clean_sql_deepseek_r1_70b,
            "mistralapi": clean_sql_mistral,
            "qwen2.5_72b": clean_sql_qwen2_5_72b
            
        }
        # print(predictions)
        
        # E. 再次清洗(可選)，讓 final_sql 更乾淨
        # final_sql_clean = extract_sql(final_sql_voted, llm="codellama")
        # 若想用其他 llm 方式清洗，可自行替換
        # final_sql_clean = extract_sql(final_sql_voted, llm="gpt")

        # # F. 判斷是否正確
        is_correct_dict = {}
        for experts,sql in predictions.items():
            gold_sql = sample.get("gold_sql")
            is_correct = False
            if gold_sql:
                # 也可先對 gold_sql 做同樣的 extract_sql 以確保大小寫或符號一致
                # gold_sql_clean = extract_sql(gold_sql, llm="codellama")
                # is_correct = (final_sql_clean.strip().lower() == gold_sql_clean.strip().lower())
                # Evaluate predictions
                db = "./data/spider/database"
                etype = "all"
                table = "./data/spider/tables.json"
                kmaps = build_foreign_key_map_from_json(table)   
                if cc:
                    is_correct = evaluate_cc(gold_sql, [sql], db, etype, kmaps)
                    is_correct_dict[experts] = is_correct
            # # G. 更新專家權重（若該輪答錯，就衰減 chosen_experts）
                    wma.update_weights(experts, is_correct)
                # current_weight = wma.get_expert_weight(experts)
            # C. 根據更新後的權重進行加權投票
            final_sql, chosen_experts, best_weight = wma.weighted_majority_vote(predictions)
            # # H. 紀錄結果
        results.append({
            "index": index+start_num_prompts,
            "question": sample["question"],
            # "experts_model": experts,
            "gold_sql":  gold_sql,
            "predicted_sql": predictions,
            "final_sql": final_sql,
            "chosen_experts": chosen_experts,
            "is_correct": is_correct_dict,
            "current_weights": wma.get_weights()
        })
        # 保存最終結果到 JSON 結構
        final_results.append({
            "index": index+start_num_prompts,
            "chosen_expert": chosen_experts,
            "best_weight":best_weight,
            "final_sql": final_sql,
        })
    final_results_path = os.path.join(path_generate, "final_result_cc.json" if cc else "final_result_no_cc.json")
    append_json(final_results_path, final_results)
    print(f"Voted results successfully appended to {final_results_path}")

    output_file = os.path.join(path_generate, "results.json_cc" if cc else "results_no_cc.json")
    append_json(output_file, results)
    print(f"Results successfully appended to {output_file}")

    return final_results,results
    

if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Call LLM on prompts and output results.")
    parser.add_argument("--path_generate", type=str, help="Path to the generated raw file.")
    parser.add_argument("--out_file", type=str, default="raw_cc.txt", help="Output file name.")
    parser.add_argument("--gold", type=str, help="Path to gold SQL.")
    parser.add_argument("--pool", type=int, default=1, help="Number of threads in the pool (default: 1).")
    parser.add_argument("--start_num_prompts", type=int, default=0)
    parser.add_argument("--end_num_prompts", type=int, default=1034, help="Number of prompts to process (default: all).")
    parser.add_argument("--call_mode", type=str, default="write", help="Mode to write or append.")


    args = parser.parse_args()
    print("Arguments received:", args)
    gold_sql = "./data/spider/dev_gold.sql"
    path_generate = "data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034_1"
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
    wma = WeightedMajorityAlgorithm(epsilon=0.005)
    # wma.add_expert(f"gptapi4", init_weight=1.0)
    # wma.add_expert("gptapi4o", init_weight=1.0)
    # wma.add_expert("o1-preview", init_weight=1.0)
    # wma.add_expert("o3-mini", init_weight=1.0)
    wma.add_expert("llamaapi_3.3", init_weight=1.0)
    wma.add_expert("qwen2.5-coderaapi", init_weight=1.0)
    # wma.add_expert("deepseekapi_r1_70b", init_weight=1.0)
    wma.add_expert("mistralapi", init_weight=1.0)
    wma.add_expert("qwen2.5_72b", init_weight=1.0)
    # # 執行多專家投票 + WMA流程
    final_results,results = run_sql_generation_wma(input_data,wma,path_generate,start_num_prompts,end_num_prompts,cc = False)

    
"""
python src/sources/wma/cc.py \
    --path_generate data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034 \
    --gold ./data/spider/dev_gold.sql \
    --start_num_prompts 0 \
    --end_num_prompts 1 \
    --call_mode "append"


"""    

