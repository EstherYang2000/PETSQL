#!/bin/bash

set -e  # 碰到錯誤就停


# === 設定參數 ===
PATH_GENERATE="bird/process/bird/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534"
START_NUM_PROMPTS=1381
END_NUM_PROMPTS=1534
N_SAMPLES=1
DATASET_TYPE="dev"
CALL_MODE="append"
ROUND=1
GOLD_SQL_PATH="./bird/bird/dev.sql"
QUESTIONS_JSON="${PATH_GENERATE}/questions.json"
EXPERTS="gemini"


# echo "## Start first round prompt generation ..."
# python src/sources/sql_gen/prompt_gen.py \
#   --data_type bird \
#   --dataset ppl_dev_bird.json \
#   --dataset_type dev \
#   --n 1534 \
#   --kshot 9 \
#   --select_type Euclidean_mask

# Step 1: 第一輪 SQL 生成 + WMA
# echo "## Start first round SQL generation with WMA ..."
# python src/sources/wma/cc_bird.py \
#     --path_generate ${PATH_GENERATE} \
#     --gold ${GOLD_SQL_PATH} \
#     --start_num_prompts ${START_NUM_PROMPTS} \
#     --end_num_prompts ${END_NUM_PROMPTS} \
#     --n_samples ${N_SAMPLES} \
#     --dataset_type ${DATASET_TYPE} \
#     --call_mode ${CALL_MODE} \
#     --refinement \
#     --rounds ${ROUND} \
#     --experts ${EXPERTS}

# python src/sources/bird_evaluation/process_sql.py \
#     --file bird/process/bird/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534/final_result_1.json \
#     --output bird/process/bird/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534/final_result_1_output_eval.json \
#     --type rf

# python src/sources/bird_evaluation/evaluation.py \
#     --predicted_sql_path bird/process/bird/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534/final_result_1_output_eval.json \
#     --ground_truth_path bird/bird/dev.sql \
#     --db_root_path bird/bird/database/ \
#     --num_cpus 4 \
#     --meta_time_out 30 \
#     --diff_json_path bird/bird/dev.json \
#     --sql_dialect SQLite \
#     --output_log_path bird/process/bird/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534/evaluation_log.txt


# Step 2: Schema Linking
# echo "## Performing Schema Linking ..."
# SQL_OUTPUT_FILE="bird/process/bird/PPL_DEV_BIRD.JSON-9_SHOT_Euclidean_mask_1534/final_sql_1_grokapi_grok-3-beta_cc.txt"

# python src/sources/schemalink.py \
#     --output ppl_${DATASET_TYPE}_add_sl.json \
#     --file ${SQL_OUTPUT_FILE} \
#     --dataset_type ${DATASET_TYPE}


# # # # # Step 3: 生成新的Prompt (帶Schema Linking資訊的Prompt)
# echo "## Generating second round prompt (with schema linking)..."
# python src/sources/sql_gen/prompt_gen.py \
#     --data_type bird \
#     --dataset ppl_dev_add_sl_bird.json \
#     --dataset_type dev \
#     --n 1534 \
#     --kshot 9 \
#     --select_type Euclidean_mask

# # # # # # # Step 4: 第二輪 SQL 生成 (帶Schema Linking版 Prompt)
echo "## Start second round SQL generation with WMA ..."

ROUND=2
PATH_GENERATE="bird/process/bird/PPL_DEV_ADD_SL_BIRD.JSON-9_SHOT_Euclidean_mask_1534"
python src/sources/wma/cc_bird.py \
    --path_generate ${PATH_GENERATE} \
    --gold ${GOLD_SQL_PATH} \
    --start_num_prompts ${START_NUM_PROMPTS} \
    --end_num_prompts ${END_NUM_PROMPTS} \
    --n_samples ${N_SAMPLES} \
    --dataset_type ${DATASET_TYPE} \
    --call_mode ${CALL_MODE} \
    --refinement \
    --rounds ${ROUND}\
    --experts ${EXPERTS}

# python src/sources/bird_evaluation/process_sql.py \
#     --file bird/process/bird/PPL_DEV_ADD_SL_BIRD.JSON-9_SHOT_Euclidean_mask_1534/final_result_1.json \
#     --output bird/process/bird/PPL_DEV_ADD_SL_BIRD.JSON-9_SHOT_Euclidean_mask_1534/final_result_1_output_eval.json \
#     --type rf

# python src/sources/bird_evaluation/evaluation.py \
#     --predicted_sql_path bird/process/bird/PPL_DEV_ADD_SL_BIRD.JSON-9_SHOT_Euclidean_mask_1534/final_result_1_output_eval.json \
#     --ground_truth_path bird/bird/dev.sql \
#     --db_root_path bird/bird/database/ \
#     --num_cpus 4 \
#     --meta_time_out 30 \
#     --diff_json_path bird/bird/dev.json \
#     --sql_dialect SQLite \
#     --output_log_path bird/process/bird/PPL_DEV_ADD_SL_BIRD.JSON-9_SHOT_Euclidean_mask_1534/evaluation_log.txt
# # # 最後提醒
# echo "✅ All steps finished! Results saved in ${PATH_GENERATE}/results.json and final_result.json"
