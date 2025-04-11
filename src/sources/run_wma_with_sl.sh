#!/bin/bash

set -e  # 碰到錯誤就停


# === 設定參數 ===
PATH_GENERATE="data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034"
START_NUM_PROMPTS=0
END_NUM_PROMPTS=1034
N_SAMPLES=1
DATASET_TYPE="dev"
DATASET="spider"
CALL_MODE="append"
ROUND=1
STRATEGY="rwma"
GOLD_SQL_PATH="./data/spider/dev_gold.sql"
QUESTIONS_JSON="${PATH_GENERATE}/questions.json"

# python src/sources/data_preprocess.py --dataset ${DATASET} --type dev


# echo "## Start first round prompt generation ..."
# python src/sources/sql_gen/prompt_gen.py \
#     --dataset ppl_${DATASET_TYPE}.json \
#     --dataset_type ${DATASET_TYPE} \
#     --n 1034 \
#     --kshot 9 \
#     --select_type Euclidean_mask

# Step 1: 第一輪 SQL 生成 + WMA
# echo "## Start first round SQL generation with WMA ..."
# python src/sources/wma/cc.py \
#     --path_generate ${PATH_GENERATE} \
#     --gold ${GOLD_SQL_PATH} \
#     --start_num_prompts ${START_NUM_PROMPTS} \
#     --end_num_prompts ${END_NUM_PROMPTS} \
#     --n_samples ${N_SAMPLES} \
#     --dataset_type ${DATASET_TYPE} \
#     --call_mode ${CALL_MODE} \
#     --refinement \
#     --rounds ${ROUND} \
#     --strategy ${STRATEGY} \
#     --auto_epsilon \


# #Step 2: Schema Linking
# echo "## Performing Schema Linking ..."
# SQL_OUTPUT_FILE="${PATH_GENERATE}/final_sql_${ROUND}.txt"

python src/sources/schemalink.py \
    --output ppl_${DATASET_TYPE}_add_sl.json \
    --file data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034_gemini_rf/final_sql_1.txt \
    --dataset_type ${DATASET_TYPE}


# # # # # # # # # # Step 3: 生成新的Prompt (帶Schema Linking資訊的Prompt)
echo "## Generating second round prompt (with schema linking)..."
python src/sources/sql_gen/prompt_gen.py \
    --dataset ppl_${DATASET_TYPE}_add_sl.json \
    --dataset_type ${DATASET_TYPE} \
    --n 1034 \
    --kshot 9 \
    --select_type Euclidean_mask \
    --sl

# # # # # Step 4: 第二輪 SQL 生成 (帶Schema Linking版 Prompt)
echo "## Start second round SQL generation with WMA ..."

ROUND=2
PATH_GENERATE="data/process/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034"
python src/sources/wma/cc.py \
    --path_generate ${PATH_GENERATE} \
    --gold ${GOLD_SQL_PATH} \
    --start_num_prompts ${START_NUM_PROMPTS} \
    --end_num_prompts ${END_NUM_PROMPTS} \
    --n_samples ${N_SAMPLES} \
    --dataset_type ${DATASET_TYPE} \
    --call_mode ${CALL_MODE} \
    --refinement \
    --rounds ${ROUND} \
    --strategy naive \
    --auto_epsilon

# # 最後提醒
echo "✅ All steps finished! Results saved in ${PATH_GENERATE}/results.json and final_result.json"
