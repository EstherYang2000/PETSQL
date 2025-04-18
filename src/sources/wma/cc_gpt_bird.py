import json
import os
from itertools import zip_longest
from wma import WeightedMajorityAlgorithm,auto_select_epsilon
from utils.file_utils import load_prompts, append_json,write_txt
from bird_evaluation.evaluation_utils import load_json,execute_sql
from bird_evaluation.evaluation import calculate_ex
from sql_gen.sql_utils import run_sql_generation, run_refinement_pipeline
import argparse


def apply_schema_linking(sql_output_file, output_sl_file):
    os.system(f"python src/sources/schemalink.py --output {output_sl_file} --file {sql_output_file}")

def run_sql_generation_wma(input_data, path_generate, start_num_prompts, end_num_prompts, dataset_type, n_samples, refinement,round=1,strategy="wma", auto_epsilon=False):
    
    expert_list = []
    expert_list.append({"name": "llamaapi_3.3", "model": "llamaapi", "version": "3.3","path":"bird/process/bird/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base/llamaapi_3.3_output.txt"})
    expert_list.append({"name": "gpt-4o", "model": "gptapi", "version": "chatgpt-4o-latest","path":"bird/process/bird/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base/gptapi_chatgpt-4o-latest_output.txt"})
    expert_list.append({"name": "gpt-4.1-2025-04-14", "model": "gptapi", "version": "gpt-4.1-2025-04-14","path":"bird/process/bird/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base/gptapi_gpt-4.1-2025-04-14_output.txt"})
    expert_list.append({"name": "o3-mini", "model": "gptapi", "version": "o3-mini","path":"bird/process/bird/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base/gptapi_o3-mini_output.txt"})
    expert_list.append({"name": "qwen_api_32b-instruct-fp16", "model": "qwen_api", "version": "32b-instruct-fp16","path":"bird/process/bird/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base/qwenapi_32b-instruct-fp16_output.txt"})
    expert_list.append({"name": "qwen_api_2_5_72b", "model": "qwen_api", "version": "2_5_72b","path":"bird/process/bird/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base/qwen_api_2_5_72b_output.txt"})
    expert_list.append({"name": "gemini", "model": "googlegeminiapi", "version": "gemini-2.5-pro-exp-03-25","path":"bird/process/bird/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base/googlegeminiapi_gemini-2.5-pro-exp-03-25_output.txt"})
    expert_list.append({"name": "grok3", "model": "grokapi", "version": "grok-3-beta","path":"bird/process/bird/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base/grokapi_grok-3-beta_output.txt"})
        # 計算 epsilon
    if auto_epsilon:
        epsilon = auto_select_epsilon(len(expert_list), end_num_prompts - start_num_prompts)
        print(f"[auto_epsilon] epsilon selected: {epsilon:.6f}")
    else:
        epsilon = 0.0005
        
    wma = WeightedMajorityAlgorithm(epsilon=epsilon)
    for expert in expert_list:
        path = expert['path']
        with open(path) as f:
            raw_sql_outputs = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]
        expert['raw_sql_outputs'] = raw_sql_outputs
        print(f"Expert: {expert['name']}, Number of raw SQL outputs: {len(raw_sql_outputs)}")
    
    for expert in expert_list:
        wma.add_expert(expert["name"], init_weight=1.0)
    results, final_results = [], []

    for index, sample in enumerate(input_data):
        predictions = {}
        if dataset_type == "dev":
            db_path = f"./bird/bird/database/{sample['db']}/{sample['db']}.sqlite"
        else:
            db_path = f"./bird/bird/test_database/{sample['db']}/{sample['db']}.sqlite"
        for expert in expert_list:
            raw_sql_output = [{
                "prompt_index":  index,
                "sql_candidates": expert['raw_sql_outputs'][index]
                    
                }]
            # raw_sql_output = run_sql_generation(
            #     model=expert["model"],
            #     prompts=[sample['prompt']],
            #     path_generate=path_generate,
            #     out_file=f"{expert['model']}_{expert['version']}_cc.json",
            #     end_num_prompts=end_num_prompts,
            #     call_mode="append",
            #     model_version=expert['version'],
            #     n_samples=n_samples,
            #     question_index=start_num_prompts + index
            # )

            if refinement:
                refined_candidates = run_refinement_pipeline(
                    db_path, sample['prompt'], raw_sql_output, path_generate, start_num_prompts + index, expert['model'], expert['version']
                )
                refined_candidates[0]['sql_candidates'] = list(set(refined_candidates[0]['sql_candidates']))
                predictions[expert['name']] = refined_candidates[0]['sql_candidates']
            else:
                predictions[expert['name']] = expert['raw_sql_outputs'][index]
        gold_data = sample.get("gold_sql")
        gold_sql, db_name = gold_data[0],gold_data[1]
        print(f"gold_sql: {gold_sql}")
        print(f"db_name: {db_name}")
        # STEP 1: 先根據目前權重做加權投票（預測）
        if strategy == "rwma":
            final_sql, chosen_experts, best_weight = wma.randomized_weighted_majority_vote(predictions)
        else:
            final_sql, chosen_experts, best_weight = wma.weighted_majority_vote(predictions)

        # STEP 2: 計算正確與否，並更新權重
        if gold_sql:
            diff_json_path = "./bird/bird/dev.json" if dataset_type == "dev" else "./bird/bird/test.json"
            diff_contents = load_json(diff_json_path)
            
            is_correct_any = False
            for expert, sql_list in predictions.items():
                is_correct_any = False  # reset for each expert
                for idx, candidate_sql in enumerate(sql_list):
                    db_info = diff_contents[idx]  # 假設idx對應題目
                    sql_dialect = "SQLite"  # SQLite, MySQL, PostgreSQL
                    # db_name = db_info["db_id"]
                    # true_gold = db_info["query"]
                    if sql_dialect == "SQLite":
                        db_path = f"./bird/bird/database/{db_name}/{db_name}.sqlite"
                    else:
                        db_path = None  # MySQL與PostgreSQL會透過連線函數連線，無需路徑
                    
                    # 使用 execute_sql 並傳入 calculate_ex 評估結果
                    res = execute_sql(
                        candidate_sql,
                        gold_sql,
                        db_path,
                        sql_dialect,
                        calculate_ex
                    )

                    if res == 1:
                        is_correct_any = True
                        break  # 只要任一SQL候選通過即視為正確

                wma.update_weights(expert, is_correct_any, strategy=strategy)
        if auto_epsilon and index > 0:
            mistake_counts = wma.get_mistake_counts()
            best_expert_name, best_mistake_count = min(mistake_counts.items(), key=lambda x: x[1])

            print(f"[Round {index}] epsilon updated to {epsilon:.6f} using best_expert: {best_expert_name} (mistakes: {best_mistake_count})")
        else:
            best_expert_name, best_mistake_count = "-", 0
        print(f"✅ Overall best expert: {best_expert_name} with {best_mistake_count} mistakes.")

        results.append({
            "index": index + start_num_prompts,
            "question": sample["question"],
            "gold_sql": gold_sql,
            "final_sql": final_sql,
            "chosen_experts": chosen_experts,
            # "is_correct": is_correct_any,
            "current_weights": wma.get_weights(),
            "current_epsilon": epsilon,
            "currenrt_mistakes": wma.get_mistake_counts(),
            "best_expert": best_expert_name,
            "best_expert_mistakes": best_mistake_count
            
        })
        final_results.append({
            "index": index + start_num_prompts,
            "final_sql": final_sql,
            "chosen_expert": chosen_experts,
            "best_weight": best_weight,
            "current_epsilon": epsilon,
            "currenrt_mistakes": wma.get_mistake_counts(),
            "best_expert": best_expert_name,
            "best_expert_mistakes": best_mistake_count
        })

    append_json(os.path.join(path_generate, f"final_result_{round}.json"), final_results)
    append_json(os.path.join(path_generate, f"results_{round}.json"), results)

    # convert to txt
    final_sql_statements = [entry["final_sql"] for entry in final_results]
    file_txt = os.path.join(path_generate, f"final_sql_{round}.txt")
    write_txt(file_txt, final_sql_statements)
    print(f"Final SQLs written to {file_txt}")
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Call LLM on prompts and output results.")
    parser.add_argument("--path_generate", type=str,
                        help="Path to the generated raw file.")
    parser.add_argument("--dataset_type", type=str,
                        help="",default="dev")
    parser.add_argument("--gold", type=str, default="bird/bird/dev.sql")
    parser.add_argument("--start_num_prompts", type=int, default=0)
    parser.add_argument("--end_num_prompts", type=int, default=1534,
                        help="Number of prompts to process from the prompt file (if not specified, take all).")
    parser.add_argument("--call_mode", type=str, default="write",
                        help="mode to write or append")
    parser.add_argument("--n_samples", type=int, default=1,
                        help="Number of samples to generate for each prompt.")
    parser.add_argument("--refinement", action="store_true",
                    help="whether to do refinement")
    parser.add_argument("--rounds",type=int, default=1)
    parser.add_argument("--strategy", type=str, default="wma", choices=["wma", "rwma","naive"],
                    help="Voting strategy to use: wma (Weighted Majority) or rwma (Randomized WMA)")

    parser.add_argument("--auto_epsilon", action="store_true",
                        help="whether to use auto epsilon")
    parser.add_argument('--experts',
                    nargs='+',  # <--- 接受一個或多個值
                    required=False,
                    help="List of experts...")
    args = parser.parse_args()
    
    path_generate = args.path_generate
    start_num_prompts = args.start_num_prompts
    end_num_prompts = args.end_num_prompts
    n_samples = args.n_samples
    dataset_type = args.dataset_type
    refinement = args.refinement
    round = args.rounds
    gold = args.gold

    with open(gold) as f:
        glist = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]

    question_path = os.path.join(path_generate, "questions.json")
    with open(question_path) as f:
        questions = json.load(f)

    all_prompts = load_prompts(path_generate, start_num_prompts, end_num_prompts)
    input_data = [
        {"prompt": p, "gold_sql": g, "question": q['question'], "db": q['db']}
        for p, g, q in zip_longest(all_prompts, glist[start_num_prompts:end_num_prompts], questions[start_num_prompts:end_num_prompts])
    ]
    # first round
    run_sql_generation_wma(input_data, path_generate, start_num_prompts, end_num_prompts, dataset_type, n_samples, refinement,round=1,strategy=args.strategy, auto_epsilon=args.auto_epsilon)



    """
    python src/sources/wma/cc_gpt_bird.py \
        --path_generate bird/process/vote/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534_base_wma \
        --gold bird/bird/dev.sql \
        --start_num_prompts 0 \
        --end_num_prompts 1534 \
        --n_samples 1 \
        --dataset_type dev \
        --call_mode append \
        --rounds 1 \
        --strategy wma \
        --auto_epsilon            

python src/sources/bird_evaluation/process_sql.py \
    --file bird/process/vote/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534_rf_naive/final_result_1.json \
    --output bird/process/vote/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534_rf_naive/final_result_output_eval.json \
    --type rf

python src/sources/bird_evaluation/evaluation.py \
    --predicted_sql_path bird/process/vote/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534_rf_naive/final_result_output_eval.json \
    --ground_truth_path bird/bird/dev.sql \
    --db_root_path bird/bird/database/ \
    --num_cpus 4 \
    --meta_time_out 30 \
    --diff_json_path bird/bird/dev.json \
    --sql_dialect SQLite \
    --output_log_path bird/process/vote/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534_rf_naive/evaluation_log.txt    
    
    
    
    
    
    
    
    
    """