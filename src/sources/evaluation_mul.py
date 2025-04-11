import json
import os
import sqlite3
from typing import List, Dict, Tuple
from process_sql import get_sql, Schema, get_schema
from evaluation import Evaluator, eval_exec_match, build_foreign_key_map_from_json, rebuild_sql_val, rebuild_sql_col, build_valid_col_units
# from wma import WeightedMajorityAlgorithm  # Assuming your WMA class is in wma.py
import argparse

def evaluate_single_question_with_candidates(
    gold_entry: Tuple[str, str],
    pred_entry: Dict,
    db_dir: str,
    kmaps: Dict,
    etype: str = "all"
) -> Dict:
    """
    Evaluate a single question with multiple SQL candidates against a gold SQL.
    
    Args:
        gold_entry (Tuple[str, str]): Gold SQL and database name (e.g., ("SELECT count(*) FROM singer", "singer_db")).
        pred_entry (Dict): Predicted entry with prompt index and SQL candidates (e.g., {"prompt_index": 0, "sql_candidates": [...]}).
        db_dir (str): Directory containing the database files.
        kmaps (Dict): Foreign key mappings for the databases.
        etype (str): Evaluation type ("all", "exec", "match").
    
    Returns:
        Dict: Evaluation results including whether any candidate is correct, scores for each candidate, and best candidate.
    """
    # Extract gold SQL and database
    g_str, db_name = gold_entry
    db_path = os.path.join(db_dir, db_name, db_name + ".sqlite")
    schema = Schema(get_schema(db_path))
    g_sql = get_sql(schema, g_str)

    # Rebuild gold SQL for value and foreign key evaluation
    kmap = kmaps[db_name]
    g_valid_col_units = build_valid_col_units(g_sql['from']['table_units'], schema)
    g_sql = rebuild_sql_val(g_sql)
    g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)

    # Extract predicted SQL candidates
    prompt_index = pred_entry["prompt_index"]
    sql_candidates = pred_entry["sql_candidates"]

    # Initialize evaluator
    evaluator = Evaluator()
    results = {
        "prompt_index": prompt_index,
        "gold_sql": g_str,
        "candidates": [],
        "any_correct": False,
        "best_candidate": None,
        "best_exact_score": 0.0,
        "best_exec_score": False
    }
    sql_candidates_set = set(sql_candidates)
    # Evaluate each candidate
    for idx, p_str in enumerate(sql_candidates_set):
        candidate_result = {
            "sql": p_str,
            "exact_score": 0.0,
            "exec_score": False,
            "partial_scores": None
        }

        # Parse predicted SQL
        try:
            p_sql = get_sql(schema, p_str)
        except Exception as e:
            print(f"Prompt {prompt_index}, Candidate {idx} - Error parsing SQL: {e}")
            candidate_result["error"] = str(e)
            results["candidates"].append(candidate_result)
            continue

        # Rebuild predicted SQL for value and foreign key evaluation
        p_valid_col_units = build_valid_col_units(p_sql['from']['table_units'], schema)
        p_sql = rebuild_sql_val(p_sql)
        p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)

        # Evaluate execution match
        if etype in ["all", "exec"]:
            exec_score = eval_exec_match(db_path, p_str, g_str, p_sql, g_sql)
            candidate_result["exec_score"] = exec_score

        # Evaluate exact match
        if etype in ["all", "match"]:
            exact_score = evaluator.eval_exact_match(p_sql, g_sql)
            partial_scores = evaluator.partial_scores
            candidate_result["exact_score"] = exact_score
            candidate_result["partial_scores"] = partial_scores

            # Print mismatch details if exact match fails
            if exact_score == 0:
                print(f"Prompt {prompt_index}, Candidate {idx} (exact match failed):")
                print(f"  Predicted: {p_str}")
                print(f"  Gold: {g_str}")
                print(f"  Partial Scores: {partial_scores}")
                print("")

        # Update overall results
        results["candidates"].append(candidate_result)
        if (etype in ["all", "exec"] and candidate_result["exec_score"]):
            results["any_correct"] = True

        # Track the best candidate based on exact match score
        if candidate_result["exact_score"] > results["best_exact_score"]:
            results["best_exact_score"] = candidate_result["exact_score"]
            results["best_candidate"] = candidate_result
        elif candidate_result["exact_score"] == results["best_exact_score"]:
            # If tied, prefer the one with exec_score=True (if applicable)
            if etype in ["all", "exec"] and candidate_result["exec_score"]:
                if not results["best_candidate"] or not results["best_candidate"]["exec_score"]:
                    results["best_candidate"] = candidate_result

    return results

def evaluate_with_wma(
    gold_entry: Tuple[str, str],
    pred_entry: Dict,
    db_dir: str,
    kmaps: Dict,
    wma: 'WeightedMajorityAlgorithm',
    etype: str = "all"
) -> Dict:
    """
    Evaluate a single question with multiple SQL candidates using WMA to select the best candidate.
    
    Args:
        gold_entry (Tuple[str, str]): Gold SQL and database name.
        pred_entry (Dict): Predicted entry with prompt index and SQL candidates.
        db_dir (str): Directory containing the database files.
        kmaps (Dict): Foreign key mappings for the databases.
        wma (WeightedMajorityAlgorithm): Instance of WMA to aggregate votes.
        etype (str): Evaluation type ("all", "exec", "match").
    
    Returns:
        Dict: Evaluation results including WMA-selected candidate and correctness.
    """
    # Evaluate all candidates
    eval_results = evaluate_single_question_with_candidates(gold_entry, pred_entry, db_dir, kmaps, etype)

    # Prepare predictions for WMA (group candidates by expert)
    prompt_index = pred_entry["prompt_index"]
    predictions_dict = {f"expert_{i}": [candidate["sql"]] for i, candidate in enumerate(eval_results["candidates"])}

    # Use WMA to select the best SQL
    best_sql, chosen_experts, best_weight = wma.weighted_majority_vote(predictions_dict)
    eval_results["wma_result"] = {
        "best_sql": best_sql,
        "chosen_experts": chosen_experts,
        "best_weight": best_weight
    }

    # Find the candidate that matches the WMA-selected SQL
    for candidate in eval_results["candidates"]:
        if candidate["sql"] == best_sql:
            eval_results["wma_selected_candidate"] = candidate
            eval_results["wma_correct"] = (etype in ["all", "exec"] and candidate["exec_score"]) or \
                                          (etype in ["all", "match"] and candidate["exact_score"] == 1)
            break
    else:
        eval_results["wma_selected_candidate"] = None
        eval_results["wma_correct"] = False

    # Update WMA weights based on correctness
    for expert in predictions_dict.keys():
        is_correct = any(
            candidate["sql"] == best_sql and (
                (etype in ["all", "exec"] and candidate["exec_score"]) or
                (etype in ["all", "match"] and candidate["exact_score"] == 1)
            )
            for candidate in eval_results["candidates"]
            if candidate["sql"] in predictions_dict[expert]
        )
        wma.update_weights(expert, is_correct)

    return eval_results
def evaluate_dataset_with_candidates(
    gold_file: str,
    pred_file: str,
    db_dir: str,
    table_file: str,
    etype: str = "all",
    use_wma: bool = False,
    num_questions: int = 1034,
    output_file: str = "correct_or_one_sqls.txt"
) -> Dict:
    """
    Evaluate a dataset with multiple SQL candidates and export either a correct SQL or one incorrect SQL to a TXT file.
    
    Args:
        output_file (str): Path to the output TXT file for SQLs.
    """
    with open(gold_file) as f:
        glist = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]

    with open(pred_file) as f:
        pred_data = json.load(f)
        plist = pred_data

    if num_questions:
        glist = glist[:num_questions]
        plist = plist[:num_questions]
    print(f"Evaluating {len(glist)} questions...")
    print(f"Gold SQLs: {len(glist)}, Predicted SQLs: {len(plist)}")

    kmaps = build_foreign_key_map_from_json(table_file)

    all_results = []
    total_questions = len(glist)
    correct_any = 0

    # Open the output file for writing SQLs
    with open(output_file, 'w') as out_f:
        for idx, (gold_entry, pred_entry) in enumerate(zip(glist, plist)):
            print(f"Evaluating question {idx + 1}/{total_questions} (Prompt Index: {pred_entry['prompt_index']})...")
            
            result = evaluate_single_question_with_candidates(gold_entry, pred_entry, db_dir, kmaps, etype)

            if result["any_correct"]:
                correct_any += 1
                # Write the best candidate SQL (highest exact_score or exec_score=True)
                best_candidate = result["best_candidate"]
                out_f.write(f"{best_candidate['sql']}\t{gold_entry[1]}\n")
            else:
                # If no correct candidate, write the first candidate
                if result["candidates"]:
                    out_f.write(f"SELECT\n")

            all_results.append(result)

    summary = {
        "total_questions": total_questions,
        "correct_any": correct_any,
        "accuracy_any": correct_any / total_questions if total_questions > 0 else 0.0,
        "results": all_results
    }

    print("\n=== Evaluation Summary ===")
    print(f"Total Questions: {summary['total_questions']}")
    print(f"Correct (Any Candidate): {summary['correct_any']}")
    print(f"Accuracy (Any Candidate): {summary['accuracy_any']:.3f}")
    print(f"SQLs (correct or one incorrect) exported to: {output_file}")

    return summary

# Main script
if __name__ == "__main__":
    # Paths and configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', dest='gold', type=str)
    parser.add_argument('--pred', dest='pred', type=str)
    parser.add_argument('--db', dest='db', type=str)
    parser.add_argument('--table', dest='table', type=str)
    parser.add_argument('--etype', dest='etype', type=str)
    parser.add_argument('--nums', dest='nums', type=int, default=1034)
    parser.add_argument('--output', dest='output', type=str, default="correct_sqls.txt")
    args = parser.parse_args()

    gold_file = args.gold
    pred_file = args.pred
    db_dir = args.db
    table_file = args.table
    etype = args.etype
    nums = args.nums
    

    use_wma = False  # Set to False if you don't want to use WMA
    output_file = "data/process/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034/llamaapi_3.3_output.txt"
    # Evaluate the dataset
    summary = evaluate_dataset_with_candidates(
        gold_file,
        pred_file,
        db_dir,
        table_file,
        etype,
        use_wma,
        nums,
        output_file
    )

    # # Optionally, save the detailed results to a file
    # with open("evaluation_results.json", "w") as f:
    #     json.dump(summary, f, indent=2)
        
        
"""
python src/sources/evaluation_mul.py \
    --gold ./data/spider/dev_gold.sql  \
    --pred data/process/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034/llamaapi_3.3_output.json \
    --etype all \
    --db ./data/spider/database \
    --table ./data/spider/tables.json \
    --num 1034

"""