import json
from evaluation import build_foreign_key_map_from_json, evaluate_cc
import argparse
def calculate_final_result(gold_path:str,final_path:str, dataset_type:str):
    # Load the gold SQL
    with open(gold_path) as f:
        glist = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]

    # Load the candidate SQL
    with open(final_path, "r", encoding="utf-8") as f:
        final_data = json.load(f)

    

    table = "./data/spider/tables.json" if dataset_type == "dev" else "./data/spider/test_tables.json"
    kmaps = build_foreign_key_map_from_json(table)
    db = "./data/spider/database" if dataset_type == "dev" else "./data/spider/test_database"
    count = 0
    final_result = []
    for idx, (gold_data, candidate_data) in enumerate(zip(glist, final_data)):
        print(f"idx: {idx}, gold_data: {gold_data}, candidate_data: {candidate_data}")
        if not evaluate_cc(gold_data, [candidate_data['final_sql']], db, "all", kmaps):
            count += 1
        candidate_data['current_error'] = count
        final_result.append(candidate_data)

    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predicted_sql_path", type=str, required=True, default=""
    )
    parser.add_argument("--ground_truth_path", type=str, required=True, default="")
    parser.add_argument("--dataset_type", type=str, required=True, default="")
    args = parser.parse_args()

    calculate_final_result(
        args.ground_truth_path,
        args.predicted_sql_path,
        args.dataset_type
    )
    
    """
    
    python src/sources/evaluation_final.py \
    --predicted_sql_path data/process/vote/202504/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034_base_wma/final_result_1.json \
    --ground_truth_path ./data/spider/dev_gold.sql \
    --dataset_type dev
    """
    