import json
import os
from itertools import zip_longest
from wma import WeightedMajorityAlgorithm
from utils.file_utils import load_prompts, append_json,write_txt
from evaluation import build_foreign_key_map_from_json, evaluate_cc
from sql_gen.sql_utils import run_sql_generation, run_refinement_pipeline
import argparse


def apply_schema_linking(sql_output_file, output_sl_file):
    os.system(f"python src/sources/schemalink.py --output {output_sl_file} --file {sql_output_file}")

def run_sql_generation_wma(input_data, path_generate, start_num_prompts, end_num_prompts, dataset_type, n_samples, refinement,round=1):
    wma = WeightedMajorityAlgorithm(epsilon=0.005)
    expert_list = [
        {"name": "llamaapi_3.3", "model": "llamaapi", "version": "3.3"},
    #     {"name": "gpt-4", "model": "gptapi", "version": "gpt-4"},
    #     {"name": "gpt-4o", "model": "gptapi", "version": "gpt-4o"},
    #     {"name": "o1-preview", "model": "gptapi", "version": "o1-preview"},
    #     {"name": "o3-mini", "model": "gptapi", "version": "o3-mini"},
    #     {"name": "qwen_api_32b-instruct-fp16", "model": "qwen_api", "version": "32b-instruct-fp16"},
    #     {"name": "mistralapi_small_24b", "model": "mistralapi", "version": "small_24b"},
    #     {"name": "qwen_api_2_5_72b", "model": "qwen_api", "version": "2_5_72b"},
    ]
    for expert in expert_list:
        wma.add_expert(expert["name"], init_weight=1.0)

    results, final_results = [], []

    for index, sample in enumerate(input_data):
        predictions = {}
        db_path = f"./data/spider/database/{sample['db']}/{sample['db']}.sqlite"
        for expert in expert_list:
            raw_sql_output = run_sql_generation(
                model=expert["model"],
                prompts=[sample['prompt']],
                path_generate=path_generate,
                out_file=f"{expert['model']}_{expert['version']}_cc.json",
                end_num_prompts=end_num_prompts,
                call_mode="append",
                model_version=expert['version'],
                n_samples=n_samples,
                question_index=start_num_prompts + index
            )

            if refinement:
                refined_candidates = run_refinement_pipeline(
                    db_path, sample['prompt'], raw_sql_output, path_generate, end_num_prompts, expert['model'], expert['version']
                )
                refined_candidates[0]['sql_candidates'] = list(set(refined_candidates[0]['sql_candidates']))
                predictions[expert['name']] = refined_candidates[0]['sql_candidates']

        gold_sql = sample.get("gold_sql")
        if gold_sql:
            table = "./data/spider/tables.json" if dataset_type == "dev" else "./data/spider/test_tables.json"
            kmaps = build_foreign_key_map_from_json(table)
            db = "./data/spider/database" if dataset_type == "dev" else "./data/spider/test_database"

            is_correct_any = False
            for expert, sql_list in predictions.items():
                for candidate_sql in sql_list:
                    if evaluate_cc(gold_sql, [candidate_sql], db, "all", kmaps):
                        is_correct_any = True
                        break
                wma.update_weights(expert, is_correct_any)

        final_sql, chosen_experts, best_weight = wma.weighted_majority_vote(predictions)
        results.append({
            "index": index + start_num_prompts,
            "question": sample["question"],
            "gold_sql": gold_sql,
            "final_sql": final_sql,
            "chosen_experts": chosen_experts,
            "is_correct": is_correct_any,
            "current_weights": wma.get_weights()
        })
        final_results.append({
            "index": index + start_num_prompts,
            "final_sql": final_sql,
            "chosen_expert": chosen_experts,
            "best_weight": best_weight
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
    parser.add_argument("--gold", type=str, default="data/spider/dev_gold.sql")
    parser.add_argument("--start_num_prompts", type=int, default=0)
    parser.add_argument("--end_num_prompts", type=int, default=1034,
                        help="Number of prompts to process from the prompt file (if not specified, take all).")
    parser.add_argument("--call_mode", type=str, default="write",
                        help="mode to write or append")
    parser.add_argument("--n_samples", type=int, default=1,
                        help="Number of samples to generate for each prompt.")
    parser.add_argument("--refinement", action="store_true",
                    help="whether to do refinement")
    parser.add_argument("--rounds",type=int, default=1)
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
    run_sql_generation_wma(input_data, path_generate, start_num_prompts, end_num_prompts, dataset_type, n_samples, refinement,round=1)

    

