# main.py (示意)
import json

from wma import WeightedMajorityAlgorithm
from post_process import extract_sql  # 假設在 post_process.py 中定義
from evaluation import evaluate_cc,build_foreign_key_map_from_json
from sql_gen.call_llm import run_sql_generation,load_prompts
import argparse
from itertools import zip_longest
import sqlparse
import os
# 初始化 WMA
wma = WeightedMajorityAlgorithm(epsilon=0.2)

# 新增專家 (模型)
wma.add_expert("codellamaapi", init_weight=1.0)
wma.add_expert("puyuapi", init_weight=1.0)
wma.add_expert("llamaapi", init_weight=1.0)


def call_expert(expert_name: str, prompt:str,path_generate:str,model_version=None) -> str:
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
        return run_sql_generation(model=expert_name,prompts=[prompt],path_generate=path_generate ,out_file="codellama.txt",pool_num=1,call_mode="append")
    elif expert_name == "puyuapi":
        return run_sql_generation(model=expert_name,prompts=[prompt],path_generate=path_generate, out_file="puyuma.txt",pool_num=1,call_mode="append")
    elif expert_name == "llamaapi":
        return run_sql_generation(model=expert_name,prompts=[prompt], path_generate=path_generate,out_file="llm.txt",pool_num=1,call_mode="append")
    elif expert_name == "gptapi":
        if model_version == "gpt-3.5-turbo":
            return run_sql_generation(model=expert_name,prompts=[prompt], path_generate=path_generate,out_file=f"{model_version}.txt",pool_num=1,call_mode="append",model_version="gpt-3.5-turbo")
        elif model_version == "gpt-4":
            return run_sql_generation(model=expert_name,prompts=[prompt], path_generate=path_generate,out_file=f"{model_version}.txt",pool_num=1,call_mode="append",model_version="gpt-4")
        elif model_version == "gpt-4o":
            return run_sql_generation(model=expert_name,prompts=[prompt], path_generate=path_generate,out_file=f"{model_version}.txt",pool_num=1,call_mode="append",model_version="gpt-4o")
        elif model_version == "o1-preview":
            return run_sql_generation(model=expert_name,prompts=[prompt], path_generate=path_generate,out_file=f"{model_version}.txt",pool_num=1,call_mode="append",model_version="o1-preview")
    else:
        return run_sql_generation(model=expert_name,prompts=[prompt], path_generate=path_generate,out_file="model.txt",pool_num=1,call_mode="append")

def run_sql_generation_wma(input_data,path_generate):
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
    

    # 初始化 Weighted Majority Algorithm
    wma = WeightedMajorityAlgorithm(epsilon=0.2)

    # 新增專家 (此處先寫死三位，如有更多可自行增加)
    # wma.add_expert("codellamaapi", init_weight=1.0)
    # wma.add_expert("puyuapi", init_weight=1.0)
    # wma.add_expert("llamaapi", init_weight=1.0)
    wma.add_expert("gptapi35", init_weight=1.0)
    wma.add_expert(f"gptapi4", init_weight=1.0)
    wma.add_expert("gptapi4o", init_weight=1.0)
    wma.add_expert("o1-preview", init_weight=1.0)
    # Add validation
    if not input_data:
        raise ValueError("input_data cannot be empty")
        
    for sample in input_data:
        if 'prompt' not in sample or not sample['prompt']:
            raise ValueError(f"Invalid sample, missing or empty prompt: {sample}")
    results = []
    final_results = []

    for index,sample in enumerate(input_data):
        # A. 各專家輸出 raw SQL
        # raw_sql_codellama = call_expert("codellamaapi", sample['prompt'],path_generate)
        raw_sql_gpt35     = call_expert("gptapi", sample['prompt'],path_generate,model_version="gpt-3.5-turbo")
        raw_sql_gpt4     = call_expert("gptapi", sample['prompt'],path_generate,model_version="gpt-4")
        raw_sql_gpt4o     = call_expert("gptapi", sample['prompt'],path_generate,model_version="gpt-4o")
        raw_sql_gpto1preview     = call_expert("gptapi", sample['prompt'],path_generate,model_version="o1-preview")
        
        # raw_sql_llama     = call_expert("llamaapi", sample['prompt'],path_generate)

        # B. 每位專家的 raw SQL 都先做基礎清洗
        # clean_sql_codellama = extract_sql(raw_sql_codellama, llm="codellama")
        clean_sql_gpt35 = sqlparse.format(extract_sql(raw_sql_gpt35, "sensechat").strip(), reindent=False)
        clean_sql_gpt4 = sqlparse.format(extract_sql(raw_sql_gpt4, "sensechat").strip(), reindent=False)
        clean_sql_gpt4o = sqlparse.format(extract_sql(raw_sql_gpt4o, "sensechat").strip(), reindent=False)
        clean_sql_gpto1preview = sqlparse.format(extract_sql(raw_sql_gpto1preview, "sensechat").strip(), reindent=False)
        
        # clean_sql_llama     = extract_sql(raw_sql_llama,     llm="llama")

        # C. 建立投票用 predictions dict
        predictions = {
            # "codellamaapi": clean_sql_codellama,
            "gptapi35":      clean_sql_gpt35,
            "gptapi4":      clean_sql_gpt4,
            "gptapi4o":      clean_sql_gpt4o,
            "o1-preview":      clean_sql_gpto1preview,
            
            # "llamaapi":     clean_sql_llama
        }
        # print(predictions)
        
        # E. 再次清洗(可選)，讓 final_sql 更乾淨
        # final_sql_clean = extract_sql(final_sql_voted, llm="codellama")
        # 若想用其他 llm 方式清洗，可自行替換
        # final_sql_clean = extract_sql(final_sql_voted, llm="gpt")

        # # F. 判斷是否正確
        is_correct_dict = {}
        for experts,sql in predictions.items():
            print(experts,sql)
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
                is_correct = evaluate_cc(gold_sql, [sql], db, etype, kmaps)
                is_correct_dict[experts] = is_correct
        # # G. 更新專家權重（若該輪答錯，就衰減 chosen_experts）
                wma.update_weights(experts, is_correct)
                # current_weight = wma.get_expert_weight(experts)
            # C. 根據更新後的權重進行加權投票
            final_sql, chosen_experts, best_weight = wma.weighted_majority_vote(predictions)
            # # H. 紀錄結果
            results.append({
                "index": index,
                "question": sample["question"],
                # "experts_model": experts,
                "gold_sql":  gold_sql,
                "predicted_sql": predictions,
                "final_sql": final_sql,
                "chosen_experts": chosen_experts,
                "is_correct": is_correct_dict,
                "current_weights": wma.get_weights()
            })
            print(results)
            # 保存最終結果到 JSON 結構
            final_results.append({
                "index": index,
                "chosen_expert": chosen_experts,
                "best_weight":best_weight,
                "final_sql": final_sql,
            })

            # results.append({
            #     "question": sample["question"],
            #     "raw_sql_codellama": raw_sql_codellama,
            #     "raw_sql_gpt":      raw_sql_gpt,
            #     "raw_sql_llama":     raw_sql_llama,
            #     "final_sql":         final_sql_clean,
            #     "gold_sql":          gold_sql,
            #     "is_correct":        is_correct,
            #     "chosen_experts":    chosen_experts,
            #     "current_weights":   wma.get_weights().copy()
            # })
            # Define the output JSON file path
        final_results_path = os.path.join(path_generate, "final_result.json")
        with open(final_results_path, "w") as f:
            json.dump(final_results, f, indent=4)
        print(f"Voted results successfully written to {final_results_path}")
        output_file = os.path.join(path_generate,"results.json")
        # Write results to a JSON file
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results successfully written to {output_file}")

    return results
    

if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Call LLM on prompts and output results.")
    parser.add_argument("--path_generate", type=str, help="Path to the generated raw file.")
    parser.add_argument("--out_file", type=str, default="raw_cc.txt", help="Output file name.")
    parser.add_argument("--gold", type=str, help="Path to gold SQL.")
    parser.add_argument("--pool", type=int, default=1, help="Number of threads in the pool (default: 1).")
    parser.add_argument("--num_prompts", type=int, default=None, help="Number of prompts to process (default: all).")
    parser.add_argument("--call_mode", type=str, default="write", help="Mode to write or append.")


    args = parser.parse_args()
    print("Arguments received:", args)
    gold_sql = "./data/spider/dev_gold.sql"
    path_generate = "data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_100"
    n = 10
    with open(gold_sql) as f:
        glist = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]
    question_path = os.path.join(path_generate,"questions.json")
    with open(question_path) as f:
        questions = json.load(f)
    print(len(glist),len(questions))
    all_prompts = load_prompts(path_generate,num_prompts=None)
    input_data = []
    input_data = [{"prompt": prompt, "gold_sql": golden,"question":questions['question']} for prompt, golden,questions in zip_longest(all_prompts[:n], glist[:n],questions[:n], fillvalue=None)]
    print(len(input_data))
    # print(input_data[0])
    # # 執行多專家投票 + WMA流程
    results = run_sql_generation_wma(input_data,path_generate)

    # # 查看結果
    # for idx, r in enumerate(results, start=1):
    #     print(f"\n--- Sample {idx} ---")
    #     print("Question:", r["question"])
    #     print("Raw codellama SQL:", r["raw_sql_codellama"])
    #     print("Raw puyu SQL:     ", r["raw_sql_puyu"])
    #     print("Raw llama SQL:    ", r["raw_sql_llama"])
    #     print("Final Clean SQL:  ", r["final_sql"])
    #     print("Gold SQL:         ", r["gold_sql"])
    #     print("Correct?          ", r["is_correct"])
    #     print("Chosen Experts:   ", r["chosen_experts"])
    #     print("Current Weights:  ", r["current_weights"])
    
    
# python src/sources/wma/cc.py \
#     --path_generate data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_5 \
#     --gold ./data/spider/dev_gold.sql \
#     --num_prompts 2 \
#     --call_mode "append"
