#1. Data Preprocessing

# echo "## Data processing ..."
# START_TIME=`date +%s`
# python src/sources/data_preprocess.py --type dev
# END_TIME=`date +%s`
# EXECUTING_TIME=`expr $END_TIME - $START_TIME`
# echo "data preprocess time consume: $EXECUTING_TIME s"


# 2. First Round of SQL Generation
# echo "## Start 1st generation of prompting ..."
# START_TIME=`date +%s`
# python src/sources/sql_gen/prompt_gen.py \
#   --dataset ppl_dev.json \
#   --dataset_type dev \
#   --n 1034 \
#   --kshot 9 \
#   --select_type Euclidean_mask 
  
# echo "## Start 1st calling LLM ..."
python src/sources/sql_gen/call_llm.py \
    --path_generate data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034 \
    --model gptapi \
    --model_version chatgpt-4o-latest \
    --out_file gptapi_chatgpt-4o-latest.json \
    --dataset_type dev \
    --start_num_prompts 0 \
    --end_num_prompts 1034 \
    --batch_size 1 \
    --call_mode append \
    --n_samples 5

python src/sources/sql_gen/call_llm.py \
    --path_generate data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034 \
    --model qwen_api \
    --model_version 32b-instruct-fp16 \
    --out_file qwenapi_32b-instruct-fp16.json \
    --dataset_type dev \
    --start_num_prompts 0 \
    --end_num_prompts 1034 \
    --batch_size 1 \
    --call_mode append \
    --n_samples 5

python src/sources/post_process.py \
    --file data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034/gptapi_chatgpt-4o-latest.json \
    --output data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034/gptapi_chatgpt-4o-latest_output.json \
    --llm sensechat
# python src/sources/extract_sql_output.py \
#     --file data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034/llamaapi_3.3_output.json \
#     --output data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034/llamaapi_3.3_output.txt


python src/sources/evaluation.py \
    --gold ./data/spider/dev_gold.sql  \
    --pred data/vote/rl/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034_base_rl/final_sql_rl.txt \
    --etype all \
    --db ./data/spider/database \
    --table ./data/spider/tables.json \
    --num 1034

# sleep 1
# echo "1st round done!"
# END_TIME=`date +%s`
# EXECUTING_TIME=`expr $END_TIME - $START_TIME`
# echo "1st round time consume: $EXECUTING_TIME s"


# # # 3. schema linking for the first round
# echo "## Start schema linking for the first round ..."
python src/sources/schemalink.py --output ppl_dev_add_sl.json --file data/process/PPL_DEV.JSON-9_SHOT_Euclidean_mask_1034/gptapi_chatgpt-4o-latest_output.txt


# # 4. Second Round of SQL Generation
# echo "## Start 2nd generation of prompting ..."
# START_TIME=`date +%s`
python src/sources/sql_gen/prompt_gen.py \
    --dataset ppl_dev_add_sl.json \
    --dataset_type dev \
    --n 1034 \
    --kshot 9 \
    --select_type Euclidean_mask \
    --sl
# # echo "## Start 2nd calling LLM ..."
python src/sources/sql_gen/call_llm.py \
    --path_generate data/process/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034_qwen_72b \
    --model qwen_api \
    --model_version 2_5_72b \
    --out_file qwen_api_2_5_72b.json \
    --dataset_type dev \
    --start_num_prompts 1 \
    --end_num_prompts 1034 \
    --batch_size 1 \
    --call_mode append \
    --n_samples 1


python src/sources/post_process.py \
    --file data/process/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034_qwen_72b/qwen_api_2_5_72b.json \
    --output data/process/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034_qwen_72b/qwen_api_2_5_72b_output.json  \
    --llm sensechat

python src/sources/extract_sql_output.py \
    --file data/process/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034_qwen_72b/qwen_api_2_5_72b_output.json \
    --output data/process/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034_qwen_72b/qwen_api_2_5_72b_output.txt

python src/sources/evaluation.py \
    --gold ./data/spider/dev_gold.sql  \
    --pred data/process/vote/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034_rf_rwma/final_sql_1.txt\
    --etype all \
    --db ./data/spider/database \
    --table ./data/spider/tables.json \
    --num 1034


# python src/sources/evaluation.py \
#     --gold ./data/spider/dev_gold.sql  \
#     --pred data/process/PPL_DEV_ADD_SL.JSON-9_SHOT_Euclidean_mask_1034/final_sql_1.txt \
#     --etype all \
#     --db ./data/spider/database \
#     --table ./data/spider/tables.json \
#     --num 1034



